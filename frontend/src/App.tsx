import { useCallback, useRef, useState } from "react";
import "./App.css";
import { FisheryPanel } from "./components/FisheryPanel";

const FISHERY_IDS = (
  (import.meta.env.VITE_FISHERY_IDS as string | undefined) ?? "fishery_a,fishery_b"
).split(",");

interface Banner {
  id: number;
  message: string;
}

const BANNER_LIFETIME_MS = 6000;

function App() {
  const [banners, setBanners] = useState<Banner[]>([]);
  const nextBannerId = useRef(0);

  // Both panels' migration events funnel through here -- this is the
  // "impossible to miss" mechanism build spec §9 asks for, on top of the
  // highlighted line each panel's own event feed already shows.
  const handleMigration = useCallback((message: string) => {
    const id = nextBannerId.current++;
    setBanners((prev) => [...prev, { id, message }]);
    window.setTimeout(() => {
      setBanners((prev) => prev.filter((b) => b.id !== id));
    }, BANNER_LIFETIME_MS);
  }, []);

  return (
    <main className="app">
      <h1>Generative Fishery</h1>

      <div className="migration-banners">
        {banners.map((banner) => (
          <div key={banner.id} className="migration-banner">
            {banner.message}
          </div>
        ))}
      </div>

      <div className="fishery-grid">
        {FISHERY_IDS.map((fisheryId) => (
          <FisheryPanel key={fisheryId} fisheryId={fisheryId} onMigration={handleMigration} />
        ))}
      </div>
    </main>
  );
}

export default App;
