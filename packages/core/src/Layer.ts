import { Module } from "./Module";
import { Neuron } from "./Neuron";
import { Value } from "./Value";

export class Layer extends Module {
  public neurons: Neuron[];
  constructor(public numInputs: number, public numOutputs: number) {
    super();
    this.neurons = Array.from(
      { length: numOutputs },
      () => new Neuron(numInputs)
    );
  }

  call(input: Value[] | number[]) {
    const out = this.neurons.map((n) => n.call(input));
    return out;
  }

  parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters());
  }
}
