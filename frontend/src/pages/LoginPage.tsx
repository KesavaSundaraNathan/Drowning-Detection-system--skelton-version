import { useState, type FormEvent } from "react";

import { useAuth } from "../authModel";
import { CenteredScreen } from "../components/CenteredScreen";
import { errorText } from "../hooks";

export default function LoginPage() {
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    login(username, password)
      .catch((err: unknown) => {
        setError(errorText(err));
      })
      .finally(() => {
        setBusy(false);
      });
  };

  return (
    <CenteredScreen subtitle="Pool Safety System">
    <div className="login">
      <form onSubmit={submit} aria-label="Log in">
        <h1 className="mb-1 text-2xl">Sign in</h1>
        <p className="small mb-2">Use your own account: the audit log records who acknowledged each alarm.</p>
        <label>
          Username
          <input autoComplete="username" value={username} onChange={(e) => { setUsername(e.target.value); }} required />
        </label>
        <label>
          Password
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => { setPassword(e.target.value); }}
            required
          />
        </label>
        <button type="submit" className="btn-primary" disabled={busy}>
          {busy ? "Logging in..." : "Log in"}
        </button>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
      </form>
    </div>
    </CenteredScreen>
  );
}
