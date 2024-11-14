import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { ExternalLink } from "lucide-react";

const SearchResults = ({
  results,
}: {
  results: { title: string; url: string; score: number; abstract: string }[];
}) => {
  //   const results = [
  //     {
  //       "title": "Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision",
  //       "url": "https://arxiv.org/pdf/2010.06775",
  //       "score": 1.325334,
  //       "abstract": "Humans learn language by listening, speaking, writing, reading, and also, via interaction with the multimodal real world..."
  //     },
  //     // ... other results would be passed as props in a real implementation
  //   ];

  return (
    <div className=" text-slate-100 p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {/* {results.length > 0 && (
          <h1 className="text-2xl font-bold mb-6 text-center text-slate-200">
            Results
          </h1>
        )} */}

        {/* <ScrollArea className="h-[800px] pr-4"> */}
        <div className="space-y-6">
          {results.map((result, index) => (
            <Card
              key={index}
              className="bg-black border-slate-00 hover:border-slate-600 transition-colors"
            >
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <a
                      href={result.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[#99c3ff] hover:text-blue-300 hover:underline text-lg font-semibold flex items-center group"
                    >
                      {result.title}
                      <ExternalLink className="ml-2 h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </a>

                    <div className="flex items-center mt-1 space-x-2">
                      <span className="text-[#bdc6c1] text-sm">
                        {result.url.replace("https://", "")}
                      </span>
                      <span className="text-slate-500 text-sm">
                        • Score: {result.score.toFixed(2)}
                      </span>
                    </div>

                    <p className="mt-2 text-[#aeaeae] line-clamp-3">
                      {result.abstract}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        {/* </ScrollArea> */}
      </div>
    </div>
  );
};

export default SearchResults;
