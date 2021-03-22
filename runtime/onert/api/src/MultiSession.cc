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

#include "MultiSession.h"
#include "nnfw_api_internal.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

multi_session::multi_session() {};

multi_session::~multi_session()
{
    while (sessions.empty()) {
        std::pair<nnfw_session *, nnfw_session *> session = sessions.back();
        sessions.pop_back();
        free(session.first);
        free(session.second);
    }
};

NNFW_STATUS multi_session::add_to_multisession(const char *package_file_path, const char *backends)
{
    NNFW_STATUS ret;
    nnfw_session *new_session;

    ret = nnfw_create_session(&new_session);
    if (ret != NNFW_STATUS_NO_ERROR) return ret;

    ret = nnfw_load_model_from_file(new_session, package_file_path);
    if (ret != NNFW_STATUS_NO_ERROR) return ret;

    ret = nnfw_set_available_backends(new_session, backends);
    if (ret != NNFW_STATUS_NO_ERROR) return ret;

    ret = nnfw_prepare(new_session);
    if (ret != NNFW_STATUS_NO_ERROR) return ret;

    sessions.push_back({new_session, NULL});
    return ret;
};

NNFW_STATUS multi_session::multisession_run_sync(int session_num)
{
    return sessions[session_num].first->run();
}

NNFW_STATUS multi_session::multisession_set_input(int session_num, uint32_t index, NNFW_TYPE type,
                                                  const void *buffer, size_t length)
{
    return sessions[session_num].first->set_input(index, type, buffer, length);
}

NNFW_STATUS multi_session::multisession_set_output(int session_num, uint32_t index, NNFW_TYPE type,
                                                   void *buffer, size_t length)
{
    return sessions[session_num].first->set_output(index, type, buffer, length);
}

NNFW_STATUS multi_session::make_dependency()
{
    nnfw_tensorinfo info_1;
    nnfw_tensorinfo info_2;

    for (uint32_t i = 0; i < sessions.size(); i++) {
        uint32_t output_sz = 0;
        nnfw_output_size(sessions[i].first, &output_sz);

        for (uint32_t j = 0; j < sessions.size(); j++) {
            if (i == j) continue;

            uint32_t input_sz = 0;
            nnfw_input_size(sessions[j].first, &input_sz);

            if (input_sz != output_sz) {
                continue;
            }
            else {
                uint32_t correct_cnt = 0;
                for (uint32_t idx1 = 0; idx1 < input_sz; idx1++) {
                    nnfw_output_tensorinfo(sessions[i].first, idx1, &info_1);
                    for (uint32_t idx2 = 0; idx2 < input_sz; idx2++) {
                        nnfw_input_tensorinfo(sessions[j].first, idx2, &info_2);

                        if (info_1.rank != info_2.rank) continue;
                        if (info_1.dtype != info_2.dtype) continue;

                        bool same = true;
                        for (int k = 0; k < info_1.rank; k++) {
                            if (info_1.dims[k] != info_2.dims[k]) same = false;
                        }
                        if (same) {
                            correct_cnt += 1;
                            break;
                        }
                    }
                }

                if (correct_cnt == input_sz) {
                    sessions[i].second = sessions[j].first;
                    break;
                }
            }
        }
    }
    return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS multi_session::multisession_run_async()
{
    make_dependency();

    for (uint32_t i = 0; i < sessions.size(); i++) {
        if (sessions[i].second != NULL) {
            sessions[i].first->set_next_session(sessions[i].second);
        }
    }

    for (uint32_t i = 0; i < sessions.size(); i++) {
        sessions[i].first->run_pipelining();
    }
    return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS multi_session::multisession_set_async_input(int session_num, uint32_t index, NNFW_TYPE type,
                                                        const void *buffer, size_t length)
{
    NNFW_STATUS ret;

    ret = sessions[session_num].first->create_new_async_desc(0);
    if (ret != NNFW_STATUS_NO_ERROR) return ret;

    ret = sessions[session_num].first->set_async_input(index, type, buffer, length, 0);
    if (ret != NNFW_STATUS_NO_ERROR) return ret;

    ret = sessions[session_num].first->async_input_post(0);
    return ret;
}

NNFW_STATUS multi_session::multisession_get_result(int session_num, std::vector<void *> outputs)
{
    return sessions[session_num].first->async_get_result(outputs);
}

NNFW_STATUS multi_session::multisession_set_finish(int session_num)
{
    return sessions[session_num].first->async_set_finish(0);
}

NNFW_STATUS multi_session::multisession_wait(int session_num)
{
    return sessions[session_num].first->async_wait();
}
