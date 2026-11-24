import { AnimatePresence, MotionConfig } from "motion/react";
import { useLocation, useOutlet } from "react-router-dom";
import { MutedBanner } from "./Banners";
import { Backdrop } from "./Backdrop";
import { LiveBootstrap } from "./LiveBootstrap";
import { NotMonitoringOverlay } from "./NotMonitoringOverlay";
import { AnimatedPage } from "./primitives/AnimatedPage";
import { SafetyAudio } from "./SafetyAudio";
import { Sidebar } from "./Sidebar";
import { VisualAlarm } from "./VisualAlarm";

/**
 * [Phase 9b] The v2 shell (ambient glass backdrop, collapsible sidebar, animated page transitions) around the
 * v3 safety layer, which is unchanged: live feed bootstrap, alarm / fault audio, the MUTED banner on every
 * screen, and the NOT MONITORING overlay above everything. Animations honour the OS "reduce motion" setting.
 */
export function Layout() {
  const location = useLocation();
  const outlet = useOutlet(); // captured per render, so the exiting page keeps its own content
  return (
    <MotionConfig reducedMotion="user">
      <div className="app relative flex h-screen w-full gap-4 overflow-hidden bg-background p-4 text-foreground ambient-grid">
        <LiveBootstrap />
        <SafetyAudio />
        <Backdrop />
        <Sidebar />
        <main className="content scroll-area-soft relative z-10 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1500px] pb-8 pr-2">
            {/* inside <main>: the collapsed red bar sticks to the top of the content (as a flex sibling of the sidebar it became a full-height red column) */}
            <NotMonitoringOverlay />
            <VisualAlarm />
            <header className="top">
              <MutedBanner />
            </header>
            <AnimatePresence mode="wait">
              <AnimatedPage key={location.pathname}>
                {outlet}
              </AnimatedPage>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </MotionConfig>
  );
}
