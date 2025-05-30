# pongrl
**Deep Reinforcement Learning project for CSC 521 - Spring 2025 - CSUDH: Pong from Pixels**

Hieu Nguyen - CSU Dominguez Hills - Spring 2025

This project explores the application of reinforcement learning techniques to the classic Pong game. Utilizing a policy gradient approach, we train an agent to improve its gameplay performance through trial and error. The repository includes detailed analyses of hyperparameter tuning, loss curves, reward distributions, and gradient norms, providing insights into the learning process and effectiveness of the model. Join us in understanding how machine learning can master one of the earliest arcade games!

**Dependencies:**
- Python: 3.12.9
- ALE: 0.10.2
- Gymnasium: 1.1.1
- Tensorflow: 2.16.1
- Numpy: 1.26.4

**To run the project:**
1. Install all dependencies
2. run karpathy.py file "python ./karpathy.py"

Change 'resume' value in the code to keep using already trained model or re-train the model from the beginning\
Change 'render' value in the code to enable interactive display for the environment\
After each 100 eps the results will be exported to results.txt file

**Result**

The following plot illustrates the average performance of the reinforcement learning agent over time in Pong
![image](https://github.com/user-attachments/assets/2bfcd815-a6a4-4f0d-9633-7f5615ff72a2)

2 pretrained model in the save model folder
1. for pong.py code (has not been able to beat Pong)
2. for Karpathy's code (constantly score around 14 points)

Video of the  model pre train\
![pretrain-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/dbdbc8c9-3595-492c-8342-fc40f2dcf146)

Video of the trained model (80000 eps)\
![trained-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/b48a3683-c602-4771-830b-2397cf948a21)


