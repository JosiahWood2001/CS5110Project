# CS5110Project
Intelligent Decision Making in Dynamic Environments Based on Evolutionary Game theory and Multi-Agent Reinforcement Learning

## INSTALLING
**PETTING ZOO IS NOT SUPPORTED ON WINDOWS**

If using windows, just use WSL (this is what I'm using)

With Linux, create venv and clone repository. Before running installs, you need to ensure you have **CMAKE** installed as some of the atari dependencies require it. You will also need **swig** and **zlib1g-dev** Rather than installing all at once, I did it in stages to work out any bugs in my env setup.

If using windows you can use Remote Dev with VS code in order to connect to your WSL system and edit code still using your developer tools.

PettingZoo Atari requires a few other sudo installs that aren't done with pip so I can't just throw them in requirements. I seldom use WSL so this may only be a me problem. If you get install errors with installing .whls that is probably what it is.

I had to build my own wheels fyi, if you have issues doing this let me know and I'll post them. I reverted to python 3.11 for better supported wheel building capabilities and pettingzoo support. Something to be aware of is old ALE is in an older g++ version so if you run it under verbose, there will be tons of warnings. The build will fail if you are using too modern of g++ or gcc because some of the legacy code had been phased out so gcc12 and g++12 is your best bet. This is because there are some parts of the std library that changed.

Note, python >=3.11 does not support ALE