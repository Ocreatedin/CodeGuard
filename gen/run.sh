
# python mark.py --dataset CodeXGLUE --pr 0.05
# python mark.py --dataset CodeXGLUE --pr 0.10
# python mark.py --dataset CodeXGLUE --pr 0.15

# python coprotector.py --dataset CodeSearchNet --pr 0.05
# python coprotector.py --dataset CodeSearchNet --pr 0.10
# python coprotector.py --dataset CodeSearchNet --pr 0.15
# python train.py --dataset CodeSearchNet --pr 0.0 --method None
# python val.py --dataset CodeSearchNet --pr 0.0 --method None

# python train.py --dataset CodeSearchNet --pr 0.05 --method CodeMark
python val.py --dataset CodeSearchNet --pr 0.05 --method CodeMark
# python wsr_codemark.py --dataset CodeSearchNet --pr 0.05


# python train.py --dataset CodeSearchNet --pr 0.10 --method CodeMark
# python train.py --dataset CodeSearchNet --pr 0.15 --method CodeMark
python val.py --dataset CodeSearchNet --pr 0.10 --method CodeMark
python val.py --dataset CodeSearchNet --pr 0.15 --method CodeMark


# python wsr_codemark.py --dataset CodeSearchNet --pr 0.10
# python wsr_codemark.py --dataset CodeSearchNet --pr 0.15

# python train.py --dataset CodeXGLUE --pr 0.05 --method CodeMark
# python train.py --dataset CodeXGLUE --pr 0.10 --method CodeMark
# python train.py --dataset CodeXGLUE --pr 0.15 --method CodeMark

python val.py --dataset CodeXGLUE --pr 0.05 --method CodeMark
python val.py --dataset CodeXGLUE --pr 0.10 --method CodeMark
python val.py --dataset CodeXGLUE --pr 0.15 --method CodeMark

# python wsr_codemark.py --dataset CodeXGLUE --pr 0.05
# python wsr_codemark.py --dataset CodeXGLUE --pr 0.10
# python wsr_codemark.py --dataset CodeXGLUE --pr 0.15


# python train.py --dataset CodeSearchNet --pr 0.05 --method CoProtector
# python train.py --dataset CodeSearchNet --pr 0.10 --method CoProtector
# python train.py --dataset CodeSearchNet --pr 0.15 --method CoProtector

python val.py --dataset CodeSearchNet --pr 0.05 --method CoProtector
python val.py --dataset CodeSearchNet --pr 0.10 --method CoProtector
python val.py --dataset CodeSearchNet --pr 0.15 --method CoProtector

# python wsr_coprotector.py --dataset CodeXGLUE --pr 0.05
# python wsr_coprotector.py --dataset CodeXGLUE --pr 0.10
# python wsr_coprotector.py --dataset CodeXGLUE --pr 0.15

python val.py --dataset CodeXGLUE --method None
python val.py --dataset CodeXGLUE --pr 0.05 --method Ghostmark
python val.py --dataset CodeXGLUE --pr 0.10 --method Ghostmark
python val.py --dataset CodeXGLUE --pr 0.15 --method Ghostmark

python val.py --dataset CodeXGLUE --pr 0.05 --method CoProtector
python val.py --dataset CodeXGLUE --pr 0.10 --method CoProtector
python val.py --dataset CodeXGLUE --pr 0.15 --method CoProtector

# shutdown