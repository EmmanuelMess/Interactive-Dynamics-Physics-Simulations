#  Interactive Dynamics for Physics Simulations

This is a test made for the [Constraint Based Simulator](https://github.com/EmmanuelMess/ConstraintBasedSimulator). It provides a constraint satisfaction physics simulator with automatic differentiation.
## Usage

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Running
```bash
python3 main.py
```

## Adding functionality

Generate a new `Constraint` subclass and use it in a case.
## Design

<img src="./design/class-diagram.png"/>

## Thanks
* [Interactive Dynamics](https://dl.acm.org/doi/pdf/10.1145/91394.91400) by Andrew Witkin, Michael Gleicher and William Welch
* 

## License

```text
MIT License

Copyright (c) 2023 EmmanuelMess

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
