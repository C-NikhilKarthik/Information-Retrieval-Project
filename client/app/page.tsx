import { SearchBar } from "@/components/SearchBar";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-4 md:p-24">
      <div className="w-full max-w-4xl space-y-8">
        <h1 className="text-4xl font-bold text-center">Search Interface</h1>
        <SearchBar className="w-full" />
      </div>
    </main>
  );
}
