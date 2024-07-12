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

$C_p(t)$ is a constraint function on a set of particles $p$ so that:

$$ \exists t / C_p(t) = 0 \land \exists \dot{C_p} $$

To obtain the constraint as a function of particles positions (instead of time) we use:

$$ \widetilde{C}_p(x(t)) = C_p(t) $$

We don't have the single particle $x(t)$ function analitically, but we can use an approximation (a Taylor approximation at $t = 0$):

$$ \widetilde{C}_p(\widetilde{x}(t)) \approx C_p(t) $$

$$ \widetilde{x}(t) = x + t * v + \frac{1}{2} * t^2 * a \approx x(t) $$

$$ \widetilde{x}(0) = x_t \land \dot{\widetilde{x}}(0) = v_t \land \ddot{\widetilde{x}}(0) = a_t $$

And that lets us compute the derivatives ($p_i \in p$ and $C^i$ is a constraint):

$$ C = \begin{bmatrix} C^0(0) & \cdots & C^m(0) \end{bmatrix}^T $$

$$ \dot{C} = \begin{bmatrix} 
 \frac{\partial C^0} {\partial t}(0) & \cdots & \frac{\partial C^m} {\partial t}(0)
\end{bmatrix}^T
$$

$$
J = \begin{bmatrix}
  \frac{\partial C_ {p_0}^0}{\partial x_1} & \frac{\partial C_ {p_0}^0}{\partial x_2} & \cdots & \frac{\partial C_ {p_n}^0}{\partial x_1} & \frac{\partial C_ {p_n}^0}{\partial x_2}  \\
  \vdots & & \vdots & & \vdots \\
  \frac{\partial C_ {p_0}^m}{\partial x_1} & \frac{\partial C_ {p_0}^m}{\partial x_2} & \cdots & \frac{\partial C_ {p_n}^m}{\partial x_1} & \frac{\partial C_ {p_n}^m}{\partial x_2}  \\
\end{bmatrix}
$$

$$
\dot{J} = \begin{bmatrix}
  \frac{\partial \dot{C}_ {p_0}}{\partial x_1} & \frac{\partial \dot{C}_ {p_0}}{\partial x_2} & \cdots & \frac{\partial \dot{C}_ {p_n}}{\partial x_1} & \frac{\partial \dot{C}_ {p_n}}{\partial x_2}  \\
  \vdots & & \vdots & & \vdots \\
  \frac{\partial \dot{C}_ {p_0}}{\partial x_1} & \frac{\partial \dot{C}_ {p_0}}{\partial x_2} & \cdots & \frac{\partial \dot{C}_ {p_n}}{\partial x_1} & \frac{\partial \dot{C}_ {p_n}}{\partial x_2}  \\
\end{bmatrix}
$$

With these we can compute λ such that:

$$
(J W J^T) {\lambda}^T + \dot{J} \dot{\widetilde{x}} + J W \ddot{\widetilde{x}} + k_s C + k_d \dot{C} = 0
$$

We compute using an approximate least squares method.

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
