# stl-gym
A tool for modifying reinforcement learning environments to incorporate Signal Temporal Logic (STL) specifications in the reward function.


## CONFIG setup
A configuration file is required for creating an STL-gym environment. The path to the file is used as an input when initializing the environment as follows: 
```Python
env = STLGym('path_to_config/config.yaml', env=None)
```

If the environment has already been made, it can be used as an input, but if left empty, must be specified in the config file. The major sections of the config file are explained below and an example is provided.

### ```env_name:```
If an environment object is not provided upon initialization, the registered name of the environment must be specified in the config file. For more information on registering a custom environment, we recommend reading the [OpenAI Gym](https://github.com/openai/gym) README.md file for more information.

### ```constants:```

### ```variables:```

### ```specifications:```

### Example: pendulum-v0

```yaml
env_name: Pendulum-v0
constants:
    - name: T
      type: int
      description: optional
variables:
    - name: var_name1
      location: obs
      identifier: int
    - name: var_name2
      location: info
      identifier: key
specifications:
    - name: spec1
      descriptor: optional description
      spec: spec1 = always(eventually[0:5]((var_name1 <= 5) and (var_name2 >= 3))
      weight: 0.6
    - name: spec2
      descriptor: optional description
      spec: spec2 = always((var_name2 >= 3) implies (eventually[0.5:1.5](var_name1 >= 3)))
    - name: spec3
      descriptor: optional description
      spec: spec3 = always(var_name3 > 5)
```

## Installation
