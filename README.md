# tm_nn

Trained network on Imitation Learning. Raw Dataset can be found on [kaggle](https://www.kaggle.com/datasets/catalystgma/trackmania-replays/).

The network was trained on a pair of replays `(Main Replay, Racing Line)`. It predicts the next move by considering a part of the Racing Line close to the current position of the car.

[![TM Imitation Learning](https://img.youtube.com/vi/RELUDK1qofE/0.jpg)](https://www.youtube.com/watch?v=RELUDK1qofE)

The model achieves modest results. Turns out that just giving a racing line for reference is pretty vague most of the time, resulting in many unwanted wall hits.

A Racing Line is formed as an array of the positions a car takes during a replay. Naturally, it contains unwanted information about the speed of the car during the replay. The following example shows that even with a slow replay accounting for the Racing Line (`~17s`) the Network can generate a decent time (`~12s`), even if it is worse than one paired with a more optimised Racing Line.

[![Versus slow Racing Line](https://img.youtube.com/vi/maPcwMK2HZo/0.jpg)](https://www.youtube.com/watch?v=maPcwMK2HZo)
