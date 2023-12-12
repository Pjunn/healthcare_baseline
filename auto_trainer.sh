while true
do
    pgrep -f train.py || python auto_trainer.py
    sleep 10
done