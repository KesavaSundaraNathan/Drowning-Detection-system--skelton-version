import { useQuery } from "@tanstack/react-query";

import { api } from "../api/endpoints";

export default function AuditPage() {
  const audit = useQuery({ queryKey: ["audit"], queryFn: api.audit });
  return (
    <div>
      <h1>Audit log</h1>
      <table>
        <thead>
          <tr>
            <th>When</th>
            <th>Who</th>
            <th>Action</th>
            <th>Target</th>
            <th>Outcome</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {audit.data?.map((e) => (
            <tr key={e.id}>
              <td>{new Date(e.at).toLocaleString()}</td>
              <td>
                {e.actor_username} ({e.ip})
              </td>
              <td>{e.action}</td>
              <td>
                {e.target_type} {e.target_id}
              </td>
              <td>{e.outcome}</td>
              <td className="small">{JSON.stringify(e.details)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
