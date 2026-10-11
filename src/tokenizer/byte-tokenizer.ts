export const BYTE_VOCAB_SIZE = 257;
export const EOS_TOKEN_ID = 256;

export class ByteTokenizer {
  public encode(text: string, appendEos = true): number[] {
    const bytes = Array.from(new TextEncoder().encode(text), (byte) => byte);
    if (appendEos) bytes.push(EOS_TOKEN_ID);
    return bytes;
  }

  public decode(tokens: readonly number[]): string {
    const bytes: number[] = [];
    for (const token of tokens) {
      if (!Number.isInteger(token) || token < 0 || token >= BYTE_VOCAB_SIZE) {
        throw new RangeError(`Invalid token ID: ${token}`);
      }
      if (token === EOS_TOKEN_ID) break;
      bytes.push(token);
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
  }
}
