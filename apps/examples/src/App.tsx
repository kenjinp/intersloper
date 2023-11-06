import { useEffect, useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import { MultilayerPerceptron, Neuron, Value } from "@intersloper/core";

function App() {
  const [count, setCount] = useState(0);

  const doStuff = () => {
    // const inputs = [1, 2, 3, 4, 5, 6, 7, 8];
    // const neuron = new Neuron(inputs.length);
    // console.log("forward", neuron.call(inputs).data);
    // console.log("backward", neuron.call(inputs).backward());

    const x = [2.0, 3.0, -1.0];
    const n = new MultilayerPerceptron(3, [4, 4, 1]);
    n.call(x);
    // console.log(n);

    const xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
    ];
    const ys = [1.0, -1.0, -1.0, 1.0]; // desired

    const ypred: Value[] = xs.map((x) => n.call(x));

    // grad descent
    // for (let k = 0; k < 20; k++) {
    //   const ypred: Value[] = xs.map((x) => n.call(x));
    //   const loss = ypred.reduce((acc, yout, i) => {
    //     return acc.add(
    //       (Array.isArray(yout) ? yout[0] : yout).sub(ys[i]).pow(2)
    //     );
    //   }, new Value(0));
    //   for (const p of n.parameters()) {
    //     p.grad = 0.0;
    //   }
    //   loss.backward();
    //   for (const p of n.parameters()) {
    //     console.log("before", p.data, p.grad, loss);
    //     p.data += -0.1 * p.grad;
    //     console.log("after", p.data, p.grad);
    //   }
    //   // console.log(n.parameters());
    //   // console.log(k, ypred, loss.data);
    // }
    console.log(ypred);
  };

  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if (e.key === "d") {
        doStuff();
      }
    };
    window.addEventListener("keydown", listener);
    return () => {
      window.removeEventListener("keydown", listener);
    };
  }, []);

  return (
    <>
      <div>
        <a href="https://vitejs.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  );
}

export default App;
