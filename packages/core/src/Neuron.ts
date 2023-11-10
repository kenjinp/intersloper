import { Module } from "./Module";
import { Random } from "./Random";
import { Value } from "./Value";

export class Neuron extends Module {
  public bias = new Value(0);
  public weights: Value[];
  constructor(
    public numInputs: number,
    public nonLinear: boolean = true,
    public random: Random = new Random()
  ) {
    super();
    this.weights = Array.from(
      { length: numInputs },
      () => new Value(this.random.randomInt(-1, 1))
    );
    this.bias = new Value(this.random.randomInt(-1, 1));
  }

  // pass inputs to neuron and calculate output
  call(input: Value[] | number[]): Value {
    const x = input.map((i) => (i instanceof Value ? i : new Value(i)));
    const act = this.weights
      .map((w, i) => w.mul(x[i]))
      .reduce((a, b) => a.add(b), this.bias);
    const out = this.nonLinear ? act.tanh() : act;

    return out;
  }

  // returns list of params
  parameters(): Value[] {
    return this.weights.concat(this.bias);
  }
}
