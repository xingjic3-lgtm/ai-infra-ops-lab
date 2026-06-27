from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    generated_tokens: int = 0


class ToyScheduler:
    def __init__(self, max_running_requests):
        self.max_running_requests = max_running_requests
        self.waiting = []
        self.running = []
        self.finished = []

    def submit(self, request):
        self.waiting.append(request)

    def step(self):
        while self.waiting and len(self.running) < self.max_running_requests:
            self.running.append(self.waiting.pop(0))

        scheduled = list(self.running)
        still_running = []

        for request in scheduled:
            request.generated_tokens += 1
            if request.generated_tokens == request.output_tokens:
                self.finished.append(request)
            else:
                still_running.append(request)

        self.running = still_running
        return scheduled


def main():
    scheduler = ToyScheduler(max_running_requests=2)
    scheduler.submit(Request("req_0", prompt_tokens=5, output_tokens=3))
    scheduler.submit(Request("req_1", prompt_tokens=4, output_tokens=2))
    scheduler.submit(Request("req_2", prompt_tokens=7, output_tokens=2))

    step_id = 0
    while scheduler.waiting or scheduler.running:
        scheduled = scheduler.step()
        names = [request.request_id for request in scheduled]
        running = [request.request_id for request in scheduler.running]
        finished = [request.request_id for request in scheduler.finished]

        print(f"step {step_id}: scheduled = {names}")
        print(f"step {step_id}: running after decode = {running}")
        print(f"step {step_id}: finished = {finished}")
        step_id += 1


if __name__ == "__main__":
    main()
