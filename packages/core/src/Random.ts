import Rand, { PRNG } from "rand-seed";
export class Random {
  private rand: Rand;
  constructor(seed?: string) {
    this.rand = new Rand(seed);
  }
  set seed(seed: string) {
    this.rand = new Rand(seed);
  }
  random() {
    return this.rand.next();
  }
  randomInt(min: number, max: number) {
    return Math.floor(this.rand.next() * (max - min + 1) + min);
  }
}
