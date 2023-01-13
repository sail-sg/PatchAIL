
# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import yaml
import os
import copy
from time import sleep
from subprocess import Popen
import argparse
from datetime import datetime

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment command")
    parser.add_argument("-p", "--priority", help="priority", type=str, default="low")

    args = parser.parse_args()

    command_to_run = "cd PatchIRL/PatchIRL; " + args.experiment
    now = datetime.now()
    now = now.strftime("%H:%M:%S")
    print(command_to_run)

    k8s_config_dict = None
    with open("/home/aiops/liumh/job_free_{}.yaml".format(args.priority), "r") as f:
        try:
            k8s_config_dict = yaml.safe_load(f)
            init_command = copy.copy(k8s_config_dict['spec']['template']['spec']['containers'][0]['command'])
        except yaml.YAMLError as exc:
            assert False, "load k8s templete file error: {}".format(exc)
        
    with open("k8s-%s.yaml" % now, "w") as f:
        k8s_config_dict['spec']['template']['spec']['containers'][0]['command'][-1] = command_to_run
        yaml.dump(k8s_config_dict, f, default_flow_style=None)
        f.flush()
    namespace = "liumh"
    if args.priority == "high":
        namespace = "immitation"
    command_line = "kubectl --namespace={} --cluster=kubernetes create -f {}".format(
        namespace, "k8s-%s.yaml" % now
    )
    
    command_line = command_line.split()
    print(command_line)
    p = Popen(command_line)