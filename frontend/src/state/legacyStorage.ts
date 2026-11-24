/**
 * Rename shim (Life Guard rebuild): `localStorage`/`sessionStorage` keys from the AquaGuard build were
 * `aquaguard.*`; the renamed keys are `lifeguard.*`. Call this once, before anything reads its own key, to
 * copy each old value across the first time a browser that still has one loads the new build. The old key is
 * left in place (harmless) so nothing is lost if this ever needs to run twice.
 */
const OLD_PREFIX = "aquaguard."; // legacy-name
const NEW_PREFIX = "lifeguard.";

function migrate(store: Storage): void {
  const keys: string[] = [];
  for (let i = 0; i < store.length; i += 1) {
    const k = store.key(i);
    if (k !== null && k.startsWith(OLD_PREFIX)) keys.push(k);
  }
  for (const oldKey of keys) {
    const newKey = NEW_PREFIX + oldKey.slice(OLD_PREFIX.length);
    if (store.getItem(newKey) === null) {
      const value = store.getItem(oldKey);
      if (value !== null) store.setItem(newKey, value);
    }
  }
}

export function migrateLegacyStorage(): void {
  try {
    migrate(window.localStorage);
    migrate(window.sessionStorage);
  } catch {
    // private browsing / blocked storage: nothing to migrate, nothing to break
  }
}
