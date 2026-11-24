import type { ZoneType } from "./api/types";

/** One colour per zone type, shared by the camera-zone and pool-zone editors. */
export const ZONE_COLOURS: Record<ZoneType, string> = { WATER: "#2b8cff", SHALLOW: "#2bd4c4", DECK: "#c9a227" };
