export enum Operations {
  ADD = "+",
  SUB = "-",
  MUL = "*",
  DIV = "/",
  POW = "**",
  NEG = "NEG",
  RELU = "ReLU",
  NONE = "",
}

const ensureValue = (other: Value | number): Value => {
  if (typeof other === "number") {
    return new Value(other);
  }
  return other;
};

export class Value {
  public grad: number = 0;
  private _backward: () => void = () => {};
  private _prev: Set<Value>;
  constructor(
    public data: number,
    private _children: Value[] = [],
    private _op = Operations.NONE
  ) {
    this._prev = new Set(_children);
  }

  add(_other: Value | number): Value {
    const other = ensureValue(_other);
    const out = new Value(
      this.data + other.data,
      [this, other],
      Operations.ADD
    );
    out._backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };
    return out;
  }

  mul(_other: Value | number): Value {
    const other = ensureValue(_other);
    const out = new Value(
      this.data * other.data,
      [this, other],
      Operations.MUL
    );
    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  pow(other: number): Value {
    if (typeof other !== "number")
      throw new Error("only supporting int/float powers for now");
    const out = new Value(this.data ** other, [this], Operations.POW);
    out._backward = () => {
      this.grad += other * this.data ** (other - 1) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(
      this.data < 0 ? 0 : this.data,
      [this],
      Operations.RELU
    );
    out._backward = () => {
      this.grad += out.data > 0 ? out.grad : 0;
    };
    return out;
  }

  neg(): Value {
    return this.mul(-1.0);
  }

  sub(other: Value | number) {
    // self - other
    return this.add(ensureValue(other).neg());
  }

  // other + self
  reverseAdd(other: Value | number): Value {
    return this.add(other);
  }

  reverseSub(_other: Value | number): Value {
    return ensureValue(_other).add(this.neg());
  }

  reverseMul(other: Value | number): Value {
    return this.mul(other);
  }

  trueDiv(other: Value | number): Value {
    return this.mul(ensureValue(other).pow(-1));
  }

  reverseTrueDiv(other: Value | number): Value {
    return this.trueDiv(other);
  }

  backward(): Value {
    // topological order all of the children in the graph
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build_topo = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        v._prev.forEach(build_topo);
        topo.push(v);
      }
    };
    build_topo(this);

    // go one variable at a time and apply the chain rule to get its gradient
    this.grad = 1;
    for (const v of topo.reverse()) {
      v._backward();
    }
    return this;
  }
}
