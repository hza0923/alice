# Runtime Interfaces

## Master

```python
class MasterService:
    async def register_worker(self, request: dict) -> dict: ...
    async def submit_request(self, request: dict) -> dict: ...
    async def report_finished(self, request: dict) -> dict: ...
```

## Worker

```python
class WorkerService:
    async def push_strategy(self, request: dict) -> dict: ...
    async def send_frame(self, frame: dict) -> dict: ...
```

## Executors

```python
class SleepExecutor:
    async def run(self, frame: dict, worker_plan: dict) -> dict: ...

class ShapeExecutor:
    async def run(self, frame: dict, worker_plan: dict) -> dict: ...
```
