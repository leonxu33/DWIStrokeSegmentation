# fenge

Go to src/

python data_aug.py
python stroke_seg_training.py
python stroke_seg_testing.py

Download ../data/result/output.py to local machine using scp
scp -i ~/.ssh/google_compute_engine Owner@[IP address]:../data/result/output.py ../data/result/

On local machine
python export_result.py