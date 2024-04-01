#  Interactive Dynamics for Physics Simulations

This is a test made for the [Constraint Based Simulator](https://github.com/EmmanuelMess/ConstraintBasedSimulator). It provides a constraint satisfaction physics simulator with automatic differentiation.

## Screenshots

<img src="./screenshots/1.gif"/> <img src="./screenshots/2.gif"/>

<img src="./screenshots/3.gif"/>

## Usage

### Setup
```bash
cd code
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Running
```bash
cd code
python3 main.py
```

## Adding functionality

Generate a new `Constraint` subclass and use it in a case.

## Design

<img src="./design/class-diagram.png"/>

## Math for a single particle

This explanation is meant to complement the references, please read that first. This explains
how to compute the derivates with respect to time and position.

C(t) is a constraint function so that:

$$ \exists t / C(t) = 0 \land \exists \dot{C} $$

To obtain the constraint as a function of particle position (instead of time) we use:

$$ \exists \widetilde{C}(x) / \widetilde{C}(x(t)) = C(t) $$

We don't have x(t) analitically, but we can use an approximation:

$$ \widetilde{C}(\widetilde{x}(t)) \approx C(t) $$

$$ \exists \widetilde{x}(0) \approx x(t) / \widetilde{x}(0) = x_t \land \dot{\widetilde{x}}(0) = v_t \land \ddot{\widetilde{x}}(0) = a_t$$

I use a Taylor approximation at t = 0:

$$ \widetilde{x}(t) = x + t * v + \frac{1}{2} * t^2 * a $$

And that lets us compute the derivatives:

$$ C $$

$$ \dot{C} $$

$$
J = \begin{bmatrix}
 \frac{\partial \widetilde{C}}{\partial x_1} & \frac{\partial \widetilde{C}}{\partial x_2}  \\
\end{bmatrix}
$$

$$
\dot{J} = \begin{bmatrix}
 \frac{\partial \dot{\widetilde{C}}}{\partial x_1} & \frac{\partial \dot{\widetilde{C}}}{\partial x_2}  \\
\end{bmatrix}
$$

With these we can compute the lagrangain for:

$$
(J W J^T) {\lambda}^T + \dot{J} \dot{\widetilde{x}} + J W \ddot{\widetilde{x}} + k_s C + k_d \dot{C}
$$

## Thanks
* [Interactive Dynamics](https://dl.acm.org/doi/pdf/10.1145/91394.91400) by Andrew Witkin, Michael Gleicher and William Welch
* [An Introduction to Physically Based Modeling: Constrained Dynamics](https://www.cs.cmu.edu/~baraff/pbm/constraints.pdf) by Andrew Witkin
* [Constrained dynamics](https://sites.cc.gatech.edu/classes/AY2017/cs7496_fall/slides/ConstrDyn.pdf) by Karen Liu
* [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) by The JAX Authors
* [pygame](https://www.pygame.org) by the pygame community

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
