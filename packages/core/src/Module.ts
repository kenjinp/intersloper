import { Value } from "./Value";

export class Module {
  public grad: number = 0;
  parameters(): Value[] {
    return [];
  }
  zeroGrad() {
    for (const p of this.parameters()) {
      p.grad = 0;
    }
  }
}
