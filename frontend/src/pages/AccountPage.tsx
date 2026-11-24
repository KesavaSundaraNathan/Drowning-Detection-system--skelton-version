import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/endpoints";
import { useAuth } from "../authModel";
import { MIN_PASSWORD, passwordError } from "../passwordModel";

/** My account: change my own password (POST /auth/password; every role). Other sessions are signed out. */
export default function AccountPage() {
  const { user } = useAuth();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [repeat, setRepeat] = useState("");
  const [local, setLocal] = useState<string | null>(null);
  const change = useMutation({
    mutationFn: () => api.changePassword(current, next),
    onSuccess: () => {
      setCurrent("");
      setNext("");
      setRepeat("");
    },
  });
  return (
    <div className="account">
      <h1>My account</h1>
      <p>
        Signed in as <strong>{user?.username}</strong> ({user?.role}).
      </p>
      <form
        className="card"
        aria-label="Change password"
        onSubmit={(e) => {
          e.preventDefault();
          change.reset();
          if (next.length < MIN_PASSWORD) setLocal(`The new password needs at least ${String(MIN_PASSWORD)} characters.`);
          else if (next !== repeat) setLocal("The two new passwords do not match.");
          else if (next === current) setLocal("The new password must differ from the current one.");
          else {
            setLocal(null);
            change.mutate();
          }
        }}
      >
        <h2>Change password</h2>
        <label>
          Current password
          <input type="password" autoComplete="current-password" value={current} onChange={(e) => { setCurrent(e.target.value); }} required />
        </label>
        <label>
          New password (min. {MIN_PASSWORD} characters)
          <input type="password" autoComplete="new-password" value={next} onChange={(e) => { setNext(e.target.value); }} required />
        </label>
        <label>
          Repeat new password
          <input type="password" autoComplete="new-password" value={repeat} onChange={(e) => { setRepeat(e.target.value); }} required />
        </label>
        <button type="submit" className="btn-primary" disabled={change.isPending}>
          Change password
        </button>
        {local && (
          <p role="alert" className="error">
            {local}
          </p>
        )}
        {change.isError && (
          <p role="alert" className="error">
            {passwordError(change.error)}
          </p>
        )}
        {change.isSuccess && (
          <p role="status" className="ok">
            Password changed. Every other session of this account has been signed out.
          </p>
        )}
      </form>
      <p className="small">
        Locked out? Another admin can reset your password on the Users page. If no admin can sign in, the site
        administrator runs <code>lifeguard admin reset-password</code> on the server (docs/OPERATOR_MANUAL.md).
      </p>
    </div>
  );
}
