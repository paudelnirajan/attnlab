import { describe, expect, it } from "vitest";
import { errorFrom } from "../api/client";

describe("errorFrom", () => {
  it("keeps the server's own error shape", async () => {
    const res = new Response(JSON.stringify({ error: { code: "busy", message: "the server is busy", detail: {} } }), {
      status: 503,
    });
    const e = await errorFrom(res);
    expect(e.code).toBe("busy");
    expect(e.message).toBe("the server is busy");
  });

  it("turns a proxy's HTML error page into a readable error", async () => {
    const e = await errorFrom(new Response("<html>Bad gateway</html>", { status: 502 }));
    expect(e.code).toBe("unreachable");
    expect(e.message).toMatch(/restarting or unreachable/);
  });

  it("names the status for anything else", async () => {
    const e = await errorFrom(new Response("nope", { status: 418 }));
    expect(e.code).toBe("http_error");
    expect(e.message).toMatch(/418/);
  });
});
