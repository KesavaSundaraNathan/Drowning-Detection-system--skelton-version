import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api/endpoints";
import type { Mode, Weekday } from "../api/types";
import { errorText } from "../hooks";

const DAYS: Weekday[] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export default function SchedulesPage() {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["schedules"], queryFn: api.schedules });
  const [name, setName] = useState("");
  const [days, setDays] = useState<Weekday[]>(["Mon", "Tue", "Wed", "Thu", "Fri"]);
  const [start, setStart] = useState("08:00");
  const [end, setEnd] = useState("20:00");
  const [mode, setMode] = useState<Mode>("Medium");
  const done = () => void qc.invalidateQueries({ queryKey: ["schedules"] });
  const create = useMutation({
    mutationFn: () => api.createSchedule({ name, days, start_time: start, end_time: end, monitoring_mode: mode }),
    onSuccess: done,
  });
  const remove = useMutation({ mutationFn: api.deleteSchedule, onSuccess: done });
  return (
    <div>
      <h1>Schedules</h1>
      <p>
        A zone with a schedule is monitored at the schedule's mode inside its window. Outside it, ordinary alerts are
        recorded but MUTED (shown in the MUTED banner). Submersion / sudden blackout alerts and faults are never muted.
      </p>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Days</th>
            <th>Window</th>
            <th>Mode</th>
            <th>Status</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {list.data?.map((s) => (
            <tr key={s.id}>
              <td>{s.name}</td>
              <td>{s.days.join(" ")}</td>
              <td>
                {s.start_time} - {s.end_time}
              </td>
              <td>{s.monitoring_mode}</td>
              <td>{s.status}</td>
              <td>
                <button type="button" onClick={() => { remove.mutate(s.id); }}>
                  Delete
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
        <h2>New schedule</h2>
        <label>
          Name <input value={name} onChange={(e) => { setName(e.target.value); }} required />
        </label>
        <fieldset>
          <legend>Days</legend>
          {DAYS.map((d) => (
            <label key={d} className="inline">
              <input
                type="checkbox"
                checked={days.includes(d)}
                onChange={(e) => { setDays((cur) => (e.target.checked ? [...cur, d] : cur.filter((x) => x !== d))); }}
              />
              {d}
            </label>
          ))}
        </fieldset>
        <label>
          From <input type="time" value={start} onChange={(e) => { setStart(e.target.value); }} />
        </label>
        <label>
          To <input type="time" value={end} onChange={(e) => { setEnd(e.target.value); }} />
        </label>
        <label>
          Mode
          <select value={mode} onChange={(e) => { setMode(e.target.value as Mode); }}>
            <option>Low</option>
            <option>Medium</option>
            <option>High</option>
          </select>
        </label>
        <button type="submit" className="btn-primary" disabled={!name || days.length === 0}>
          Create
        </button>
        {(create.isError || remove.isError) && (
          <p role="alert" className="error">
            {errorText(create.error ?? remove.error)}
          </p>
        )}
      </form>
    </div>
  );
}
