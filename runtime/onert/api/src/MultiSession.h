/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __API_MULTISESSION_H__
#define __API_MULTISESSION_H__

#include "../include/nnfw.h"
#include "../include/nnfw_experimental.h"
#include "../../core/include/exec/IODescription.h"

#include <util/GeneralConfigSource.h>
#include <util/TracingCtx.h>

#include <string>
#include <memory>

struct multi_session
{
public:
  multi_session();
  ~multi_session();
  NNFW_STATUS add_to_multisession(const char *package_file_path, const char *backends);
  NNFW_STATUS multisession_run_sync(int session_num);
  NNFW_STATUS multisession_set_input(int session_num, uint32_t index, NNFW_TYPE type,
                                     const void *buffer, size_t length);
  NNFW_STATUS multisession_set_output(int session_num, uint32_t index, NNFW_TYPE type,
                                      void *buffer, size_t length);
  NNFW_STATUS make_dependency();
  NNFW_STATUS multisession_run_async();
  NNFW_STATUS multisession_set_async_input(int session_num, uint32_t index, NNFW_TYPE type,
                                           const void *buffer, size_t length);
  NNFW_STATUS multisession_get_result(int session_num, std::vector<void *> outputs);

  NNFW_STATUS multisession_set_finish(int session_num); // for debugging
  NNFW_STATUS multisession_wait(int session_num); // for debugging

private:
  std::vector<std::pair<nnfw_session *, nnfw_session *>> sessions; // first is current session, second is next session
};

#endif // __API_NNFW_API_INTERNAL_H__
