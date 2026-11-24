import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api/endpoints";
import type { Role } from "../api/types";
import { errorText } from "../hooks";
import { MIN_PASSWORD, passwordError } from "../passwordModel";

export default function UsersPage() {
  const qc = useQueryClient();
  const users = useQuery({ queryKey: ["users"], queryFn: api.users });
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("lifeguard");
  const done = () => void qc.invalidateQueries({ queryKey: ["users"] });
  const create = useMutation({
    mutationFn: () => api.createUser({ username, password, role }),
    onSuccess: () => {
      setUsername("");
      setPassword("");
      done();
    },
  });
  const patch = useMutation({
    mutationFn: (v: { id: number; is_active: boolean }) => api.patchUser(v.id, { is_active: v.is_active }),
    onSuccess: done,
  });
  // admin reset of ANOTHER account (POST /users/{id}/password): every session of that user is signed out
  const [resetId, setResetId] = useState("");
  const [resetPw, setResetPw] = useState("");
  const [resetLocal, setResetLocal] = useState<string | null>(null);
  const reset = useMutation({
    mutationFn: () => api.resetPassword(Number(resetId), resetPw),
    onSuccess: () => {
      setResetPw("");
    },
  });
  return (
    <div>
      <h1>Users</h1>
      <table>
        <thead>
          <tr>
            <th>User</th>
            <th>Role</th>
            <th>Active</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {users.data?.map((u) => (
            <tr key={u.id}>
              <td>{u.username}</td>
              <td>{u.role}</td>
              <td>{u.is_active ? "yes" : "no"}</td>
              <td>
                <button type="button" onClick={() => { patch.mutate({ id: u.id, is_active: !u.is_active }); }}>
                  {u.is_active ? "Deactivate" : "Activate"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <form
        className="card"
        onSubmit={(e) => {
          e.preventDefault();
          create.mutate();
        }}
      >
        <h2>New user</h2>
        <label>
          Username <input value={username} onChange={(e) => { setUsername(e.target.value); }} required />
        </label>
        <label>
          Password (min. 12 characters) <input type="password" autoComplete="new-password" value={password} onChange={(e) => { setPassword(e.target.value); }} required />
        </label>
        <label>
          Role
          <select value={role} onChange={(e) => { setRole(e.target.value as Role); }}>
            <option value="lifeguard">lifeguard</option>
            <option value="supervisor">supervisor</option>
            <option value="admin">admin</option>
          </select>
        </label>
        <button type="submit" className="btn-primary">
          Create
        </button>
        {(create.isError || patch.isError) && (
          <p role="alert" className="error">
            {errorText(create.error ?? patch.error)}
          </p>
        )}
      </form>
      <form
        className="card"
        aria-label="Reset a password"
        onSubmit={(e) => {
          e.preventDefault();
          reset.reset();
          if (resetPw.length < MIN_PASSWORD) setResetLocal(`The new password needs at least ${String(MIN_PASSWORD)} characters.`);
          else {
            setResetLocal(null);
            reset.mutate();
          }
        }}
      >
        <h2>Reset a password</h2>
        <p className="small">For a user who forgot theirs. To change your own, use My account (your name, bottom of the sidebar).</p>
        <label>
          User
          <select value={resetId} onChange={(e) => { setResetId(e.target.value); }} required>
            <option value="">choose...</option>
            {users.data?.map((u) => (
              <option key={u.id} value={String(u.id)}>
                {u.username} ({u.role})
              </option>
            ))}
          </select>
        </label>
        <label>
          New password (min. {MIN_PASSWORD} characters)
          <input type="password" autoComplete="new-password" value={resetPw} onChange={(e) => { setResetPw(e.target.value); }} required />
        </label>
        <button type="submit" disabled={reset.isPending || resetId === ""}>
          Reset password
        </button>
        {resetLocal && (
          <p role="alert" className="error">
            {resetLocal}
          </p>
        )}
        {reset.isError && (
          <p role="alert" className="error">
            {passwordError(reset.error)}
          </p>
        )}
        {reset.isSuccess && (
          <p role="status" className="ok">
            Password reset. All of that user's sessions were signed out; give them the new password in person.
          </p>
        )}
      </form>
    </div>
  );
}
