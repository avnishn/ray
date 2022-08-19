from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.offline.json_reader import JsonReader


reader = JsonReader("/home/ray/halfcheetah_expert_sac.json")
writer = JsonWriter("/home/ray/halfcheetah_expert_sac_2.json")
num_rows = 0
for sb in reader.read_all_files():
    num_rows += 1
    if not isinstance(sb["dones"], list):
        sb["dones"] = [sb["dones"]]
    writer.write(sb)

print(num_rows)
