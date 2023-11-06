import { Layer } from "./Layer";
import { Module } from "./Module";
import { Value } from "./Value";

export class MultilayerPerceptron extends Module {
  public layers: Layer[];
  constructor(public numInputs: number, public numOutputs: number[]) {
    super();
    const size = [numInputs].concat(numOutputs);
    this.layers = size
      .slice(0, size.length - 1)
      .map((_, i) => new Layer(size[i], size[i + 1]));
  }

  call(input: Value[] | number[]) {
    let out = input;
    for (const layer of this.layers) {
      out = layer.call(out);
    }
    return out;
  }

  parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters());
  }
}
