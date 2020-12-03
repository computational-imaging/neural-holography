python main.py --channel=0 --method="$1" --root_path=./phases
python main.py --channel=1 --method="$1" --root_path=./phases
python main.py --channel=2 --method="$1" --root_path=./phases
python eval.py --channel=3 --root_path=./phases/"$1"_ASM
