import { ApiError } from "./api/client";
import { errorText } from "./hooks";

/** Server-side policy (core/security/passwords.py): >= 12 characters, not the username, not a common password. */
export const MIN_PASSWORD = 12;

/** Operator wording for a failed password change / reset. */
export function passwordError(e: unknown): string {
  if (e instanceof ApiError && e.status === 403) return "The current password is wrong.";
  if (e instanceof ApiError && e.status === 429) return "Too many attempts - wait a few minutes and try again.";
  return errorText(e).replace(/^password_policy:\s*/, "Password rejected: ");
}
