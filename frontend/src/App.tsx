/**
 * Life Guard operator UI (plan Phase 6).
 *
 *   not logged in        -> Login
 *   logged in            -> "Arm audio" gate -> monitoring (banners, NOT MONITORING overlay, routes)
 *
 * Routes are code-split (React.lazy); the initial chunk carries only the shell, the safety UI and
 * the dashboard's dependencies (budget: 250 KB gzipped, tools/check_bundle.mjs in CI).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, Suspense, type ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";

import type { Role } from "./api/types";
import { AuthProvider } from "./auth";
import { atLeast, useAuth } from "./authModel";
import { ArmAudioGate } from "./components/ArmAudioGate";
import { Layout } from "./components/Layout";
import DashboardPage from "./pages/DashboardPage";
import LoginPage from "./pages/LoginPage";
import { SettingsIndex } from "./pages/SettingsPage";

const CamerasPage = lazy(() => import("./pages/CamerasPage"));
const CameraDetailPage = lazy(() => import("./pages/CameraDetailPage"));
const AlertsPage = lazy(() => import("./pages/AlertsPage"));
const ZonesPage = lazy(() => import("./pages/ZonesPage"));
const CalibrationPage = lazy(() => import("./pages/CalibrationPage"));
const SchedulesPage = lazy(() => import("./pages/SchedulesPage"));
const SettingsPage = lazy(() => import("./pages/SettingsPage"));
const SystemPage = lazy(() => import("./pages/SystemPage"));
const UsersPage = lazy(() => import("./pages/UsersPage"));
const AuditPage = lazy(() => import("./pages/AuditPage"));
const AccountPage = lazy(() => import("./pages/AccountPage"));

function RequireRole({ role, children }: { role: Role; children: ReactNode }) {
  const { user } = useAuth();
  return atLeast(user, role) ? <>{children}</> : <Navigate to="/" replace />;
}

/** An old top-level path (bookmarks, alert links) -> its place under Settings, keeping ?query. */
function MovedToSettings({ to }: { to: string }) {
  const { search } = useLocation();
  return <Navigate to={`/settings/${to}${search}`} replace />;
}

function Gate() {
  const { user, loading } = useAuth();
  if (loading) return <p className="loading">Loading...</p>;
  if (!user) return <LoginPage />;
  return (
    <ArmAudioGate>
      <Suspense fallback={<p className="loading">Loading...</p>}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<DashboardPage />} />
            <Route path="cameras" element={<CamerasPage />} />
            <Route path="cameras/:cameraId" element={<CameraDetailPage />} />
            <Route path="alerts" element={<AlertsPage />} />
            <Route path="settings" element={<RequireRole role="supervisor"><SettingsPage /></RequireRole>}>
              <Route index element={<SettingsIndex />} />
              <Route path="zones" element={<RequireRole role="supervisor"><ZonesPage /></RequireRole>} />
              <Route path="calibration" element={<RequireRole role="supervisor"><CalibrationPage /></RequireRole>} />
              <Route path="schedules" element={<RequireRole role="supervisor"><SchedulesPage /></RequireRole>} />
              <Route path="system" element={<RequireRole role="supervisor"><SystemPage /></RequireRole>} />
              <Route path="users" element={<RequireRole role="admin"><UsersPage /></RequireRole>} />
              <Route path="audit" element={<RequireRole role="admin"><AuditPage /></RequireRole>} />
            </Route>
            {["zones", "calibration", "schedules", "system", "users", "audit"].map((p) => (
              <Route key={p} path={p} element={<MovedToSettings to={p} />} />
            ))}
            <Route path="account" element={<AccountPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </ArmAudioGate>
  );
}

export function App({ client }: { client: QueryClient }) {
  return (
    <QueryClientProvider client={client}>
      <AuthProvider>
        <BrowserRouter>
          <Gate />
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
