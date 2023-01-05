PYTHONPATH=. python ./src/main.py -do_process True -dataset samsum -save_path data/samsum_omission -preprocessing_num_workers 32
PYTHONPATH=. python ./src/main.py -do_process True -dataset data/dialogsum/dialogsum -save_path data/dialogsum_omission -preprocessing_num_workers 32
PYTHONPATH=. python ./src/main.py -do_process True -dataset data/qmsum/qmsum -save_path data/qmsum_omission -preprocessing_num_workers 32
PYTHONPATH=. python ./src/main.py -do_process True -dataset data/tweetsumm/tweetsumm -save_path data/tweetsumm_omission -preprocessing_num_workers 32
PYTHONPATH=. python ./src/main.py -do_process True -dataset data/emailsum/emailsum -save_path data/emailsum_omission -preprocessing_num_workers 32