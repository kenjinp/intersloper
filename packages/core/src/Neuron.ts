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
  call(input: Value[] | number[]) {
    const act = this.weights.reduce(
      (acc, w, i) => acc.add(w.mul(input[i])),
      this.bias
    );

    return this.nonLinear ? act.relu() : act;
  }

  // returns list of params
  parameters(): Value[] {
    return this.weights.concat(this.bias);
  }
}
