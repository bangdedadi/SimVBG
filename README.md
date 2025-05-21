# SimVBG

## Experiment Setup

### Main Experiment
Run `main_cv.py` with the following parameters:
setting = "multi_agent_voter"
use_nature_options = True
use_coordinator = True
optimize_story = False
load_stories = False

### Baseline Experiments

#### Baseline 1 (Full Info)
Run `main_cv.py` with:
setting = "origin_full"

#### Baseline 2 (RAG)
Run `topk_cv.py` with:
"topk": 3
"threshold": 0.3

### Ablation Experiments

1. **Complete System**
setting = "multi_agent_voter"
use_nature_options = True
use_coordinator = True
optimize_story = False
load_stories = False

2. **Without Story Module**
setting = "origintext_voter"
use_nature_options = False
use_coordinator = True
optimize_story = False
load_stories = True

3. **Without CAB Module**
setting = "single_answer"
use_nature_options = True
use_coordinator = True
optimize_story = False
load_stories = True

### Profile Impact Experiment
Run `profile_impact.py` and adjust `step_size` to change the number of profile information items added each time. Default `step_size = 58`.

## Important Notes

- Please adjust the language model path to your local model path. If using API services, replace `your_api_here` with your actual API key.
- The code provides parallel computing capabilities. You can adjust `max_workers` according to your needs to control the number of parallel processes.
