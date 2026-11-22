export type Device = 'default' | 'cpu' | 'cuda';
export type FinishReason = 'stop' | 'length' | 'context_full' | 'cancelled';

export interface OpenOptions {
  /** Default: CUDA when the library has it, else CPU. */
  device?: Device;
  /** 0 or omitted: the model's trained context, fitted to GPU memory. */
  contextTokens?: number;
  threads?: number;
  gpuDevice?: number;
  gpuMemoryLimitBytes?: number | bigint;
  kernelCacheDir?: string;
}

export interface GenerationOptions {
  /** 0 or omitted: until the model stops or the context is full. */
  maxTokens?: number;
  /** Reasoning for templates that support it (Qwen3). Default true. */
  thinking?: boolean;
  /** 0 (default): greedy. */
  temperature?: number;
  topP?: number;
  topK?: number;
  /** Omitted or negative: a different seed each call. */
  seed?: number;
  /** Generation ends at the first of these; it is not part of the text. */
  stop?: string[];
  /** Receives text as it is generated; return false to stop. */
  onText?: (text: string) => boolean | void;
}

export interface Message {
  role: 'system' | 'user' | 'assistant' | string;
  content: string;
}

export interface Completion {
  text: string;
  finishReason: FinishReason;
  promptTokens: number;
  generatedTokens: number;
  ttftMs: number;
  tokensPerSecond: number;
}

export class PalitoError extends Error {}

export class Palito {
  static open(modelPath: string, options?: OpenOptions): Promise<Palito>;
  readonly contextTokens: number;
  readonly profile: string;
  chat(messages: Message[], options?: GenerationOptions): Promise<Completion>;
  complete(prompt: string, options?: GenerationOptions): Promise<Completion>;
  streamChat(messages: Message[], options?: Omit<GenerationOptions, 'onText'>): AsyncGenerator<string>;
  streamComplete(prompt: string, options?: Omit<GenerationOptions, 'onText'>): AsyncGenerator<string>;
  close(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
}

export function version(): string;
export function libraryPath(): string;
