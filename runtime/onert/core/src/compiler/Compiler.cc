/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "compiler/Compiler.h"

#include "ParamChecker.h"
#include "ExecutorFactory.h"
#include "ShapeValidator.h"

#include <backend/builtin/Config.h>
#include "compiler/BackendManager.h"
#include "compiler/IScheduler.h"
#include "compiler/ManualScheduler.h"
#include "compiler/HEScheduler.h"
#include "compiler/StaticShapeInferer.h"
#include "compiler/OperationLowerInfo.h"
#include "compiler/pass/ConstantOutputPass.h"
#include "compiler/pass/OddOutputPass.h"
#include "compiler/pass/PassRunner.h"
#include "compiler/pass/UnusedOperandEliminationPass.h"
#include "exec/ExecTime.h"
#include "ir/verifier/Verifier.h"
#include "dumper/dot/DotDumper.h"
#include "compiler/Linear.h"
#include "interp/InterpExecutor.h"
#include "util/ConfigSource.h"
#include "util/logging.h"
#include "ir/OperationDumper.h"
#include "ir/OperationCloner.h"
#include "misc/string_helpers.h"

namespace
{

using namespace onert;

std::string getOpBackends(std::unordered_map<ir::OpCode, std::string> &opcode_to_backend)
{
  std::unordered_map<ir::OpCode, std::string>::iterator it;
  std::string opbackends;

  for (it = opcode_to_backend.begin(); it != opcode_to_backend.end(); ++it)
  {
    if (!opbackends.empty())
      opbackends = opbackends + ", ";

    auto opcode = it->first;
    const std::string opname = ir::toString(opcode);
    opbackends += opname + "=" + it->second;
  }
  return opbackends;
}

} // namespace

namespace onert
{

namespace compiler
{

CompilerOptions fetchCompilerOptionsFromGlobalConfig(const ir::Subgraphs &subgs)
{
  CompilerOptions options;
  options.backend_list = nnfw::misc::split(util::getConfigString(util::config::BACKENDS), ';');
  options.trace_filepath = util::getConfigString(util::config::TRACE_FILEPATH);
  options.graph_dump_level = util::getConfigInt(util::config::GRAPH_DOT_DUMP);
  options.executor = util::getConfigString(util::config::EXECUTOR);
  options.he_scheduler = util::getConfigBool(util::config::USE_SCHEDULER);
  options.he_profiling_mode = util::getConfigBool(util::config::PROFILING_MODE);
  options.disable_compile = util::getConfigBool(util::config::DISABLE_COMPILE);
  options.fp16_enable = util::getConfigBool(util::config::FP16_ENABLE);

  {
    // Backend for all
    auto &ms_options = options.manual_scheduler_options;

    // Default value for op_backend_all is first element in the backend list
    ms_options.backend_for_all = util::getConfigString(util::config::OP_BACKEND_ALLOPS);

// Opcode to Backend
#define OP(OpName)                                                                      \
  {                                                                                     \
    const auto &backend_str = util::getConfigString(util::config::OP_BACKEND_##OpName); \
    if (!backend_str.empty())                                                           \
    {                                                                                   \
      ms_options.opcode_to_backend[ir::OpCode::OpName] = backend_str;                   \
    }                                                                                   \
  }
#include "ir/Operations.lst"
#undef OP

    // Index to Backend
    // TODO Support multiple subgraphs for manual scheduling
    auto map_str = util::getConfigString(util::config::OP_BACKEND_MAP);
    auto key_val_list = nnfw::misc::split(map_str, ';');
    for (const auto &key_val_str : key_val_list)
    {
      if (key_val_str.empty())
      {
        continue;
      }

      auto key_val = nnfw::misc::split(key_val_str, '=');
      const auto &key_str = key_val.at(0);
      const auto &val = key_val.at(1);
      auto key = static_cast<uint32_t>(std::stoi(key_str));

      subgs.at(ir::SubgraphIndex{0})
        ->operations()
        .at(ir::OperationIndex{key}); // Check if exist, or this wil throw
      ms_options.index_to_backend.emplace(ir::OperationIndex{key}, val);
    }
  }
  return options;
}

Compiler::Compiler(const std::shared_ptr<ir::Subgraphs> &subgs, util::TracingCtx *tracing_ctx)
  : _subgraphs{subgs}, _state{State::CREATED}
{
  // Set default values for CompilerOptions
  // All these default values should not be fetched from Env, when we stop supporting Android NN
  // API.
  _options = fetchCompilerOptionsFromGlobalConfig(*subgs);

  _options.tracing_ctx = tracing_ctx;
}

void Compiler::enableToFp16() { _options.fp16_enable = true; }

void Compiler::checkProfilerConditions()
{
  if (!_options.he_scheduler)
    throw std::runtime_error("Heterogeneous scheduler must be enabled during profiling.");

  if (_options.executor != "Dataflow")
    throw std::runtime_error("Profiling mode works only with 'Dataflow' executor");
}

void Compiler::assignPartialGraph(std::unordered_map<ir::SubgraphIndex, ir::OperationIndex> &split_ops)
{
  std::cout << "shlee assignParitalGraph1" << std::endl;
  
  auto num_partialgraphs = split_ops.size() + 1;
  auto partialgraphs = std::make_shared<ir::Subgraphs>();

  for (uint32_t idx = 0; idx < num_partialgraphs; idx++)
  {
    auto partialgraph = std::make_unique<ir::Graph>();
    partialgraphs->push(ir::SubgraphIndex{idx}, std::move(partialgraph));
  }
  _subgraphs->primary()->setPartialgraphs(partialgraphs);

  auto whole_graph = _subgraphs->at(ir::SubgraphIndex{0});
  auto partial_graph = whole_graph->partialgraphs();

  _subgraphs->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
    // Lower: Assign backend
    auto lowered_graph = std::make_unique<compiler::LoweredGraph>(subg, _options);

    // auto lowered_graph =
    // std::make_unique<compiler::LoweredGraph>(_subgraphs->at(ir::SubgraphIndex{0}), _options);

    auto whole_op_order = lowered_graph->graph().topolSortOperations();
    uint32_t subgraph_ind = partial_graph->count() - 1;

    auto iter = split_ops.begin();
    for (auto op_ind : whole_op_order)
    {
     // std::cout << "shlee operation index : " << op_ind.value() << std::endl;

      if (iter == split_ops.end())
      {
         //std::cout << "shlee operation index" << (iter->first.value()+1) << std::endl;
        _options.manual_scheduler_options.index_to_partial[ir::OperationIndex{op_ind.value()}] =
          ir::SubgraphIndex{subgraph_ind};
        continue;
      }
      else
      {
        // std::cout << "shlee operation iter->first : " << iter->first << " iter->second : " << iter->second << std::endl;
        _options.manual_scheduler_options.index_to_partial[ir::OperationIndex{op_ind.value()}] =
          iter->first;
      }
      if (op_ind == iter->second)
      {
        iter++;
      }
    }
  });

  whole_graph->operations().iterate(
    [&](const ir::OperationIndex &op_ind, const ir::Operation &operation) {
      auto graph_ind = _options.manual_scheduler_options.index_to_partial.find(op_ind);
      if (graph_ind == _options.manual_scheduler_options.index_to_partial.end())
      {
        std::cout << "shlee operation error" << std::endl;
      }

      auto part = partial_graph->at(graph_ind->second);

      auto io_list = (operation.getInputs() + operation.getOutputs()) | ir::Remove::DUPLICATED |
                     ir::Remove::UNDEFINED;
      for (auto operand_ind : io_list)
      {
        if (part->operands().exist(operand_ind))
          continue;

        // Copy the operand and insert it to the partial graph
        const auto &operand = whole_graph->operands().at(operand_ind);

        auto new_operand = std::make_unique<ir::Operand>(operand);
        new_operand->clearDefUse();

        auto new_operand_ind = part->addOperand(operand_ind, std::move(new_operand));
        UNUSED_RELEASE(new_operand_ind);
        assert(new_operand_ind == operand_ind);
      }

      auto new_op_ind = part->addOperation(op_ind, clone(operation));
      UNUSED_RELEASE(new_op_ind);
      assert(new_op_ind == op_ind);
    });


  whole_graph->operands().iterate(
    [&](const ir::OperandIndex &operand_ind, const ir::Operand &operand) {
      auto uses = operand.getUses();

      for (auto use : uses)
      {
        auto graph_ind =
          _options.manual_scheduler_options.index_to_partial.find(use);
        if (graph_ind == _options.manual_scheduler_options.index_to_partial.end())
        {
          //std::cout << "shlee error :: " << graph_ind->second << std::endl;
          break;
        }

        auto part = partial_graph->at(graph_ind->second);

        if (part->operands().exist(operand_ind))
        {
          continue;
        }
        auto new_operand = std::make_unique<ir::Operand>(operand);
        new_operand->clearDefUse();
        // auto &partial_graph = *part;
        auto new_operand_ind = part->addOperand(operand_ind, std::move(new_operand));
        UNUSED_RELEASE(new_operand_ind);
        assert(new_operand_ind == operand_ind);
      }
    });



  for (uint32_t idx = 0; idx < partial_graph->count(); idx++)
  {
    auto part = partial_graph->at(ir::SubgraphIndex{idx});
    //part->finishBuilding();
    
    part->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &operand) {
      // Inputs are either "graph input" or "no def op and non-constant"
      if (whole_graph->getInputs().contains(ind) ||
          (!operand.getDef().valid() && !operand.isConstant()))
        // Outputs are either "graph output" or "no uses"
        part->addInput(ind);
      if (whole_graph->getOutputs().contains(ind) || operand.getUses().size() == 0)
        part->addOutput(ind);
    });
  }
}

std::vector<std::shared_ptr<exec::ExecutorMap>> Compiler::compile(void)
{
  std::cout << "shlee compile" << std::endl;
  // Set control flow backend for control flow operators
  {
    auto &builtin_id = backend::builtin::Config::ID;
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::If] = builtin_id;
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::While] = builtin_id;
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::Permute] = builtin_id;
  }

  // FIXME This is a workaround for bcq operations, should remove it
  {
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQFullyConnected] = "bcq";
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQGather] = "bcq";
  }

  {
    VERBOSE(Compiler) << std::boolalpha << "==== Compiler Options ====" << std::endl;
    VERBOSE(Compiler) << "backend_list             : "
                      << nnfw::misc::join(_options.backend_list.begin(),
                                          _options.backend_list.end(), "/")
                      << std::endl;
    VERBOSE(Compiler) << "trace_filepath           : " << _options.trace_filepath << std::endl;
    VERBOSE(Compiler) << "graph_dump_level         : " << _options.graph_dump_level << std::endl;
    VERBOSE(Compiler) << "executor                 : " << _options.executor << std::endl;
    VERBOSE(Compiler) << "manual backend_for_all   : "
                      << _options.manual_scheduler_options.backend_for_all << std::endl;
    VERBOSE(Compiler) << "manual_scheduler_options : "
                      << getOpBackends(_options.manual_scheduler_options.opcode_to_backend)
                      << std::endl;
    VERBOSE(Compiler) << "he_scheduler             : " << _options.he_scheduler << std::endl;
    VERBOSE(Compiler) << "he_profiling_mode        : " << _options.he_profiling_mode << std::endl;
    VERBOSE(Compiler) << "disable_compile          : " << _options.disable_compile << std::endl;
    VERBOSE(Compiler) << "fp16_enable              : " << _options.fp16_enable << std::endl
                      << std::noboolalpha;
  }

  std::vector<std::shared_ptr<exec::ExecutorMap>> executors;
  auto executor_map = std::make_shared<exec::ExecutorMap>();

  if (!checkPartitioning())
  {
    // Synchronous Run
    _subgraphs->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
      // Mandatory passes
      pass::PassRunner{}
        .append(std::make_unique<pass::ConstantOutputPass>(subg))
        .append(std::make_unique<pass::OddOutputPass>(subg))
        .run();
    });
  }
  else
  {
    // Asynchronous Run
    _subgraphs->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
      // Mandatory passes
      auto part = subg.partialgraphs();
      part->iterate([&](const ir::SubgraphIndex &, ir::Graph &partialgraph) {
        pass::PassRunner{}
          .append(std::make_unique<pass::ConstantOutputPass>(partialgraph))
          .append(std::make_unique<pass::OddOutputPass>(partialgraph))
          .run();
      });
    });
  }

  /***************************************************
   * Prepare compilation phase
   ***************************************************/

  // Compilable check
  // TODO: Support hybrid execution -
  //       execution between interpreter and compiled executor (including control flow)
  if (!checkCompilable())
  {
    _subgraphs->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
      executor_map->emplace(index, std::make_unique<interp::InterpExecutor>(subg));
      executors.push_back(executor_map);
    });
    _state = State::COMPILED;
    return executors;
  }

  // Mode check
  if (_options.he_profiling_mode)
    checkProfilerConditions();

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_options.graph_dump_level);

  if (!checkPartitioning())
  {
    // Lower: Assign backend
    std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>> lowered_subgs;
    _subgraphs->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
      onert::dumper::dot::DotDumper dot_dumper(subg, dump_level);
      dot_dumper.dump(nnfw::misc::str("before_lower_subg-", index.value()));

      // Lower: Assign backend
      lowered_subgs[index] = std::make_unique<compiler::LoweredGraph>(subg, _options);

      subg.setSubgraphs(nullptr);
    });

    _subgraphs.reset();

    for (auto &pair : lowered_subgs)
    {
      const auto &subg_index = pair.first;
      auto &lowered_subg = pair.second;
      onert::dumper::dot::DotDumper dot_dumper_lowered(lowered_subg.get(), dump_level);
      dot_dumper_lowered.dump("after_lower_subg-" + std::to_string(subg_index.value()));
    }

    // for (auto &pair : lowered_partialgraphs)
    // {
    //   const auto &partialgraph_index = pair.first;
    //   auto &lowered_partialgraph = pair.second;
    //   onert::dumper::dot::DotDumper dot_dumper_lowered_part(lowered_partialgraph.get(),
    //   dump_level); dot_dumper_lowered_part.dump("after_lower_partialgraph-" +
    //   std::to_string(partialgraph_index.value()));
    // }

    // Shape inference.
    {
      const auto primary_subg_idx = ir::SubgraphIndex{0};
      StaticShapeInferer inferer(primary_subg_idx, lowered_subgs);
      auto &lowered_subg = lowered_subgs.at(primary_subg_idx);
      auto ordered_ops = lowered_subg->graph().topolSortOperations();
      for (auto op_ind : ordered_ops)
      {
        // std::cout << "shlee shape inference : " << op_ind.value() << std::endl;
        const auto &op = lowered_subg->graph().operations().at(op_ind);
        bool has_dynamic_tensor = inferer.infer(op);
        lowered_subg->setHasDynamicTensor(op_ind, has_dynamic_tensor);
      }
      inferer.dump();
    }

    // Shape validation
    // TODO Move shape independent feature check from ShapeValidator to OperationValidator
    // TODO Move ShapeValidator into shape inference
    //      - Check input tensor shape validation
    //      - Check parameter value validation which valid value is depend on input tensor shape
    //      - Output tensor shape validation check is needless because
    //        static/dynamic shape inferer will make valid output shape
    for (auto &pair : lowered_subgs)
    {
      auto &lowered_subg = pair.second;
      compiler::ShapeValidator{lowered_subg->graph()}();
    }

    /*************************************************************
     *  Backend independent analysis & optimization phase finished
     *************************************************************/

    executor_map = std::make_shared<exec::ExecutorMap>();
    for (auto &pair : lowered_subgs)
    {
      const auto &subg_index = pair.first;
      auto &lowered_subg = pair.second;
      auto indexed_ranks = lowered_subg->indexed_ranks();

      ir::OperationDumper dumper("Executor generation of Subgraph " +
                                 std::to_string(subg_index.value()));
      lowered_subg->graph().operations().iterate(
        [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });
      auto executor = std::unique_ptr<exec::IExecutor>{
        ExecutorFactory::get().create(std::move(lowered_subg), _options, executor_map)};
      executor->setIndexedRanks(indexed_ranks);
      executor_map->insert(std::make_pair(subg_index, std::move(executor)));
      executors.push_back(executor_map);
    }
    
  }
  else
  {
    // Lower: Assign backend
    std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>>
      lowered_partialgraphs;
    _subgraphs->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
      auto part = subg.partialgraphs();
      std::cout << "shlee part count " << part->count() << std::endl;
      part->iterate([&](const ir::SubgraphIndex &pindex, ir::Graph &partialgraph) {
        onert::dumper::dot::DotDumper dot_dumper_part(partialgraph, dump_level);
        dot_dumper_part.dump(nnfw::misc::str("before_lower_subg_partialgraph-", pindex.value()));

        // // Lower: Assign backend
        lowered_partialgraphs[pindex] =
          std::make_unique<compiler::LoweredGraph>(partialgraph, _options);
        partialgraph.setSubgraphs(nullptr);
      });
    });

    _subgraphs.reset();

    for (auto &pair : lowered_partialgraphs)
    {
      const auto &partialgraph_index = pair.first;
      auto &lowered_partialgraph = pair.second;
      onert::dumper::dot::DotDumper dot_dumper_lowered_part(lowered_partialgraph.get(), dump_level);
      dot_dumper_lowered_part.dump("after_lower_subg_partialgraph-" +
                                   std::to_string(partialgraph_index.value()));
    }

    // Partial Graph shape inference
    for (auto &pair : lowered_partialgraphs)
    {
      const auto &partialgraph_index = pair.first;
      auto &lowered_partialgraph = pair.second;
      StaticShapeInferer partial_inferer(partialgraph_index, lowered_partialgraphs);
      auto ordered_ops = lowered_partialgraph->graph().topolSortOperations();
      for (auto op_ind : ordered_ops)
      {
        const auto &op = lowered_partialgraph->graph().operations().at(op_ind);
        bool has_dynamic_tensor = partial_inferer.infer(op);
        lowered_partialgraph->setHasDynamicTensor(op_ind, has_dynamic_tensor);
      }
      partial_inferer.dump();
    }

    // Shape validation
    // TODO Move shape independent feature check from ShapeValidator to OperationValidator
    // TODO Move ShapeValidator into shape inference
    //      - Check input tensor shape validation
    //      - Check parameter value validation which valid value is depend on input tensor shape
    //      - Output tensor shape validation check is needless because
    //        static/dynamic shape inferer will make valid output shape
    for (auto &pair : lowered_partialgraphs)
    {
      auto &lowered_partialgraph = pair.second;
      compiler::ShapeValidator{lowered_partialgraph->graph()}();
    }

    /*************************************************************
     *  Backend independent analysis & optimization phase finished
     *************************************************************/

    for (auto &pair : lowered_partialgraphs)
    {
      executor_map = std::make_shared<exec::ExecutorMap>();
      const auto &partialgraph_index = pair.first;
      auto &lowered_partialgraph = pair.second;
      auto indexed_ranks = lowered_partialgraph->indexed_ranks();

      std::cout << "partial graph index : " << partialgraph_index << std::endl;
      ir::OperationDumper dumper("Executor generation of Subgraph " +
                                 std::to_string(partialgraph_index.value()));
      lowered_partialgraph->graph().operations().iterate(
        [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });
      auto executor = std::unique_ptr<exec::IExecutor>{ExecutorFactory::get().create(
        std::move(lowered_partialgraph), _options, executor_map)};
      executor->setIndexedRanks(indexed_ranks);
      executor_map->insert(std::make_pair(ir::SubgraphIndex{0}, std::move(executor)));
      executors.push_back(executor_map);
    }
  }

  /********************************
   * Code generation phase finished
   ********************************/
  _state = State::COMPILED;
  std::cout << "shlee prepare finish" << std::endl;

  return executors;
}

bool Compiler::checkCompilable()
{
  // Disable compile phase
  // When ready to use interpreter backend, remove this config and use backend setting
  if (_options.disable_compile)
  {
    return false;
  }

  // TODO check unspecified operand shape

  // Check compilable parameter
  for (uint32_t i = 0; i < _subgraphs->count(); ++i)
  {
    auto graph = _subgraphs->at(ir::SubgraphIndex{i});
    ParamChecker paramChecker{graph};
    paramChecker();
    if (paramChecker.haveNoneConstParam())
    {
      return false;
    }
  }

  return true;
}

bool Compiler::checkPartitioning()
{
  bool b = true;
  std::cout << "shlee checkPartitioning : " << b << std::endl;

  return b;
}

void Compiler::compile_partial(void)
{
 if(!checkPartitioning()) return;

  std::unordered_map<ir::SubgraphIndex, ir::OperationIndex> split_ops = { 
    {ir::SubgraphIndex{0}, ir::OperationIndex{110}},
    //{ir::SubgraphIndex{1}, ir::OperationIndex{215}},
    };

  //split_ops.push_back(ir::SubgraphIndex{0}, ir::OperationIndex{110});
  //split_ops.push_back(ir::SubgraphIndex{1}, ir::OperationIndex{215});

  assignPartialGraph(split_ops);
}

} // namespace compiler

} // namespace onert