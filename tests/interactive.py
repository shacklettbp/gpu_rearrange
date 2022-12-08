import madrona_python
import gpu_rearrange_python
import torch
import torchvision
import sys
import termios
import tty

def get_single_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())
    
    ch = sys.stdin.read(1)
    
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return ch

sim = gpu_rearrange_python.RearrangeSimulator(
        gpu_id = 0,
        num_worlds = 1,
        render_width = 1024,
        render_height = 1024,
        episode_file = "/home/bps/rl/rearrange-data/datasets/replica_cad/rearrange/v1/train/rearrange_easy.json.gz",
        data_dir = "/home/bps/rl/rearrange-data/"
)

actions = sim.move_action_tensor().to_torch()
rgb_observations = sim.rgb_tensor().to_torch()
print(actions.shape, actions.dtype)
print(rgb_observations.shape, rgb_observations.dtype)


while True:
    sim.step()
    torchvision.utils.save_image((rgb_observations[0].float() / 255).permute(2, 0, 1), "/tmp/t.png")

    key_action = get_single_char()

    if (key_action == 'w'):
        action = 1
    elif (key_action == 'a'):
        action = 2
    elif (key_action == 'd'):
        action = 3
    else:
        print("Unknown action", key_action)
        sys.exit(1)

    actions[0][0] = action