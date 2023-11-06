import { Value } from "./Value";
import { spawn } from "child_process";

describe("Value", () => {
  describe("pytorch test cases", () => {
    it("should agree with pytorch", async () => {
      const callToPython = () =>
        new Promise<string>((resolve, reject) => {
          const pythonProcess = spawn("python3", ["src/Value.spec.py"]);
          pythonProcess.stdout.on("data", (data) => {
            resolve(data.toString());
          });
          pythonProcess.stderr.on("data", (data) => {
            reject(data.toString());
          });
        });

      const [ypt, xpt] = (await callToPython()).split("\n").filter(Boolean);

      let x = new Value(-4.0);
      let z = x.mul(2).add(2).add(x);
      let q = z.relu().add(z.mul(x));
      let h = z.mul(z).relu();
      let y = h.add(q).add(q.mul(x));
      y.backward();
      const [xmg, ymg] = [x, y];
      // # forward pass went well
      expect(ymg.data).toEqual(Number(ypt));
      // # backward pass went well
      expect(xmg.grad).toEqual(Number(xpt));
    });
  });

  describe("add", () => {
    it("should add two values", () => {
      const a = new Value(1);
      const b = new Value(2);
      const c = a.add(b);
      expect(c.data).toBe(3);
    });
    it("should backpropagate", () => {
      const a = new Value(1);
      const b = new Value(2);
      const c = a.add(b);
      c.backward();
      expect(a.grad).toBe(1);
      expect(b.grad).toBe(1);
    });
  });
  describe("mul", () => {
    it("should multiply two values", () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = a.mul(b);
      expect(c.data).toBe(6);
    });
    it("should backpropagate", () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = a.mul(b);
      c.backward();
      expect(a.grad).toBe(3);
      expect(b.grad).toBe(2);
    });
  });
  describe("pow", () => {
    it("should raise to a power", () => {
      const a = new Value(2);
      const b = a.pow(3);
      expect(b.data).toBe(8);
    });
    it("should backpropagate", () => {
      const a = new Value(2);
      const b = a.pow(3);
      b.backward();
      expect(a.grad).toBe(12);
      expect(b.grad).toBe(1);
    });
  });

  describe("relu", () => {
    it("should return the value if positive", () => {
      const a = new Value(2);
      const b = a.relu();
      expect(b.data).toBe(2);
    });
    it("should return 0 if negative", () => {
      const a = new Value(-2);
      const b = a.relu();
      expect(b.data).toBe(0);
    });
    it("should backpropagate", () => {
      const a = new Value(-2);
      const b = a.relu();
      b.backward();
      expect(a.grad).toBe(0);
      expect(b.grad).toBe(0);
    });
  });

  describe("neg", () => {
    it("should return the negative value", () => {
      const a = new Value(2);
      const b = a.neg();
      expect(b.data).toBe(-2);
    });
    it("should backpropagate", () => {
      const a = new Value(2);
      const b = a.neg();
      b.backward();
      expect(a.grad).toBe(-1);
      expect(b.grad).toBe(0);
    });
  });

  describe("sub", () => {
    it("should subtract two values", () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = a.sub(b);
      expect(c.data).toBe(-1);
    });
    it("should backpropagate", () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = a.sub(b);
      c.backward();
      expect(a.grad).toBe(1);
      expect(b.grad).toBe(-1);
    });
  });

  describe("trueDiv", () => {
    it("should divide two values", () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = a.trueDiv(b);
      expect(c.data).toBe(2 / 3);
    });
    it("should backpropagate", () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = a.trueDiv(b);
      c.backward();
      expect(a.grad).toBe(1 / 3);
      expect(b.grad).toBe(-2 / 9);
    });
  });

  describe("backward", () => {
    it("should backpropagate through a graph", () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = a.mul(b);
      const d = c.add(a);
      d.backward();
      expect(a.grad).toBe(4);
      expect(b.grad).toBe(2);
    });
  });
});
