"use client";

import { SearchBar } from "@/components/SearchBar";
import SearchResults from "@/components/SearchResults";
import { useState } from "react";

export default function Home() {
  const [data, setData] = useState([]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-4 md:p-24">
      <div className="w-full max-w-4xl">
        <h1 className="text-[clamp(80px,10vw,100px)] mt-40 leading-[1.1] font-bold text-center">
          MIRAX
        </h1>
        <h1 className="text-2xl font-bold mb-10 text-center">
          Information Retrieval Project
        </h1>
        <SearchBar className="w-full" setData={setData} />
      </div>
      <SearchResults results={data} />
    </main>
  );
}
