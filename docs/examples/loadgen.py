# Streaming chat-completions load generator for a PD-disaggregated RBG.
# Traffic goes through the router role so that both prefill and decode are
# exercised. Tune CONCURRENCY and DURATION to your cluster.
import asyncio, aiohttp, time, random, os

ROUTER = os.environ.get(
    "ROUTER",
    "http://qwen-pd-router-0.s-qwen-pd-router.default.svc.cluster.local:8000",
)
DURATION = int(os.environ.get("DURATION", "480"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "24"))
WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "hello", "world", "test", "benchmark", "kubernetes", "inference", "model",
]

async def worker(wid, session, stop):
    n = 0
    while time.time() < stop:
        isl = random.choice([1024, 1536, 2048])
        prompt = " ".join(random.choices(WORDS, k=isl))
        body = {
            "model": "/models/Qwen/Qwen3.5-4B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": True,
        }
        try:
            async with session.post(
                ROUTER + "/v1/chat/completions",
                json=body,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as r:
                async for _ in r.content:
                    pass
                n += 1
        except Exception as e:
            print(f"w{wid} err {e!r}", flush=True)
            await asyncio.sleep(1)
    print(f"worker {wid} done: {n} reqs", flush=True)

async def main():
    stop = time.time() + DURATION
    async with aiohttp.ClientSession() as s:
        await asyncio.gather(*[worker(i, s, stop) for i in range(CONCURRENCY)])

asyncio.run(main())
