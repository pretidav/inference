# RL reverse model attack 

## Model training and serving  
1. train 

python train_serve_predict.py --train   

2. serve 

python train_serve_predict.py --serve 

3. test predictions

python train_serve_predict.py --predict 

## A2C attack

python attack.py --batch=8 --episodes=50 --eps=0.05 --alpha=1 --randomstart

   
