import { MultilayerPerceptron } from "./MultilayerPerceptron";
import { Value } from "./Value";
import { zip } from "lodash";

describe.only("Neural Network", () => {
  const net = new MultilayerPerceptron(3, [4, 4, 1]);
  const inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ];
  const desiredTargets = [1.0, -1.0, -1.0, 1.0];
  // TODO the performance leaves something to be desired here...
  const range = 2000;

  let loss = new Value(0.0);
  for (let k = 0; k < range; k++) {
    // Forward Pass
    let yPredictions = inputs.map((input) => net.call(input)[0] as Value);
    const zippedTargetsAndPredictions = zip(desiredTargets, yPredictions);
    loss = new Value(0.0);
    for (const [ygt, yout] of zippedTargetsAndPredictions) {
      loss = loss.add(new Value(ygt!).sub(yout!).pow(2));
    }

    // backward pass
    for (const p of net.parameters()) {
      p.grad = 0.0;
    }
    loss.backward();

    // update
    for (const p of net.parameters()) {
      p.data += -0.1 * p.grad;
    }
  }
  const predictions = inputs
    .map((input) => net.call(input)[0] as Value)
    .map((v) => v.data);

  it("should should have reasonable loss", () => {
    expect(loss.data).toBeLessThan(0.01);
  });
  it("should predict the desired targets within a reasonable margin", () => {
    predictions.forEach((p, index) =>
      expect(desiredTargets[index] - p).toBeLessThan(0.01)
    );
  });
});
