
# # =====================================clean=====================================
# # python train.py --dataset CodeSearchNet --pr 0.0 --method None
# python val.py --dataset CodeSearchNet --pr 0.0 --method None

# # python train.py --dataset CodeXGLUE --pr 0.0 --method None
# python val.py --dataset CodeXGLUE --pr 0.0 --method None

# # =====================================Ghostmark=====================================
python coprotector.py --dataset CodeSearchNet --pr 0.05
python coprotector.py --dataset CodeSearchNet --pr 0.1
python coprotector.py --dataset CodeSearchNet --pr 0.15
python self-attention.py --dataset CodeSearchNet --pr 0.05
python self-attention.py --dataset CodeSearchNet --pr 0.1
python self-attention.py --dataset CodeSearchNet --pr 0.15
python codemark.py --dataset CodeSearchNet --pr 0.05
python codemark.py --dataset CodeSearchNet --pr 0.1
python codemark.py --dataset CodeSearchNet --pr 0.15

python coprotector.py --dataset CodeXGLUE --pr 0.05
python coprotector.py --dataset CodeXGLUE --pr 0.1
python coprotector.py --dataset CodeXGLUE --pr 0.15
python self-attention.py --dataset CodeXGLUE --pr 0.05
python self-attention.py --dataset CodeXGLUE --pr 0.1
python self-attention.py --dataset CodeXGLUE --pr 0.15
python codemark.py --dataset CodeXGLUE --pr 0.05
python codemark.py --dataset CodeXGLUE --pr 0.1
python codemark.py --dataset CodeXGLUE --pr 0.15

# # python train.py --dataset CodeSearchNet --pr 0.05 --method Ghostmark
# python val.py --dataset CodeSearchNet --pr 0.05 --method Ghostmark
# python wsr.py --dataset CodeSearchNet --pr 0.05

# # python self-attention.py --dataset CodeXGLUE --pr 0.05
# # python train.py --dataset CodeXGLUE --pr 0.05 --method Ghostmark
# python val.py --dataset CodeXGLUE --pr 0.05 --method Ghostmark
# python wsr.py --dataset CodeXGLUE --pr 0.05

# python self-attention.py --dataset CodeSearchNet --pr 0.1
# python train.py --dataset CodeSearchNet --pr 0.1 --method Ghostmark
# python val.py --dataset CodeSearchNet --pr 0.1 --method Ghostmark
# python wsr.py --dataset CodeSearchNet --pr 0.1

# python self-attention.py --dataset CodeXGLUE --pr 0.1
# python train.py --dataset CodeXGLUE --pr 0.1 --method Ghostmark
# python val.py --dataset CodeXGLUE --pr 0.1 --method Ghostmark
# python wsr.py --dataset CodeXGLUE --pr 0.1

# python self-attention.py --dataset CodeSearchNet --pr 0.15
# python train.py --dataset CodeSearchNet --pr 0.15 --method Ghostmark
# python val.py --dataset CodeSearchNet --pr 0.15 --method Ghostmark
# python wsr.py --dataset CodeSearchNet --pr 0.15

# python self-attention.py --dataset CodeXGLUE --pr 0.15
# python train.py --dataset CodeXGLUE --pr 0.15 --method Ghostmark
# python val.py --dataset CodeXGLUE --pr 0.15 --method Ghostmark
# python wsr.py --dataset CodeXGLUE --pr 0.15

# # =====================================CoProtector=====================================

# python train.py --dataset CodeSearchNet --pr 0.05 --method CoProtector
# python val.py --dataset CodeSearchNet --pr 0.05 --method CoProtector
# python wsr_coprotector.py --dataset CodeSearchNet --pr 0.05

# python coprotector.py --dataset CodeXGLUE --pr 0.05
# python train.py --dataset CodeXGLUE --pr 0.05 --method CoProtector
# python val.py --dataset CodeXGLUE --pr 0.05 --method CoProtector
# python wsr_coprotector.py --dataset CodeXGLUE --pr 0.05

# python train.py --dataset CodeSearchNet --pr 0.1 --method CoProtector
# python val.py --dataset CodeSearchNet --pr 0.1 --method CoProtector
# python wsr_coprotector.py --dataset CodeSearchNet --pr 0.1

# python coprotector.py --dataset CodeXGLUE --pr 0.1
# python train.py --dataset CodeXGLUE --pr 0.1 --method CoProtector
# python val.py --dataset CodeXGLUE --pr 0.1 --method CoProtector
# python wsr_coprotector.py --dataset CodeXGLUE --pr 0.1

# python train.py --dataset CodeSearchNet --pr 0.15 --method CoProtector
# python val.py --dataset CodeSearchNet --pr 0.15 --method CoProtector
# python wsr_coprotector.py --dataset CodeSearchNet --pr 0.15

# python coprotector.py --dataset CodeXGLUE --pr 0.15
# python train.py --dataset CodeXGLUE --pr 0.15 --method CoProtector
# python val.py --dataset CodeXGLUE --pr 0.15 --method CoProtector
# python wsr_coprotector.py --dataset CodeXGLUE --pr 0.15

# =====================================CodeMark=====================================

# python train.py --dataset CodeSearchNet --pr 0.05 --method CodeMark
# python val.py --dataset CodeSearchNet --pr 0.05 --method CodeMark
# python wsr_codemark.py --dataset CodeSearchNet --pr 0.05

# python codemark.py --dataset CodeXGLUE --pr 0.05
# python train.py --dataset CodeXGLUE --pr 0.05 --method CodeMark
# python val.py --dataset CodeXGLUE --pr 0.05 --method CodeMark
# python wsr_codemark.py --dataset CodeXGLUE --pr 0.05

# python train.py --dataset CodeSearchNet --pr 0.1 --method CodeMark
# python val.py --dataset CodeSearchNet --pr 0.1 --method CodeMark
# python wsr_codemark.py --dataset CodeSearchNet --pr 0.1

# python codemark.py --dataset CodeXGLUE --pr 0.1
# python train.py --dataset CodeXGLUE --pr 0.1 --method CodeMark
# python val.py --dataset CodeXGLUE --pr 0.1 --method CodeMark
# python wsr_codemark.py --dataset CodeXGLUE --pr 0.1

# python train.py --dataset CodeSearchNet --pr 0.15 --method CodeMark
# python val.py --dataset CodeSearchNet --pr 0.15 --method CodeMark
# python wsr_codemark.py --dataset CodeSearchNet --pr 0.15

# python codemark.py --dataset CodeXGLUE --pr 0.15
# python train.py --dataset CodeXGLUE --pr 0.15 --method CodeMark
# python val.py --dataset CodeXGLUE --pr 0.15 --method CodeMark
# python wsr_codemark.py --dataset CodeXGLUE --pr 0.15

# shutdown