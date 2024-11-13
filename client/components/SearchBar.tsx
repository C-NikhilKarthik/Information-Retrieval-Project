"use client";

import { useState } from "react";
import { Paperclip, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

interface SearchBarProps {
  className?: string;
  onSearchComplete?: (result: any) => void;
}

export function SearchBar({ className, onSearchComplete }: SearchBarProps) {
  const [text, setText] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type.startsWith("image/")) {
        setImage(file);
        toast({
          title: "Image uploaded",
          description: file.name,
        });
      } else {
        toast({
          title: "Invalid file type",
          description: "Please upload an image file",
          variant: "destructive",
        });
      }
    }
  };

  const handleSearch = async () => {
    if (!text && !image) {
      toast({
        title: "Input required",
        description: "Please provide text or upload an image",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    const formData = new FormData();

    // Determine query type
    let queryType = "text";
    if (image && !text) queryType = "image";
    if (image && text) queryType = "combined";

    formData.append("queryType", queryType);
    if (text) formData.append("text", text);
    if (image) formData.append("image", image);

    let endpoint = "";
    const baseUrl = "http://localhost:8080";

    switch (queryType) {
      case "text":
        endpoint = `${baseUrl}/Text_Query`;
        break;
      case "image":
        endpoint = `${baseUrl}/Image_Query`;
        break;
      case "combined":
        endpoint = `${baseUrl}/Combined_Query`;
        break;
      default:
        throw new Error("Invalid query type");
    }

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Search failed");

      const result = await response.json();

      if (onSearchComplete) {
        onSearchComplete(result);
      }

      toast({
        title: "Search completed",
        description: "Results have been updated",
      });
    } catch (error) {
      console.error("Search error:", error);
      toast({
        title: "Search failed",
        description: "Please try again later",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className={cn(
        "flex items-center gap-2 w-full mx-auto p-2 bg-background border rounded-full shadow-sm",
        className
      )}
    >
      <label
        htmlFor="file-input"
        className="p-2 hover:bg-muted rounded-full cursor-pointer transition-colors"
      >
        <Paperclip className="h-5 w-5" />
      </label>
      <input
        type="file"
        id="file-input"
        accept="image/*"
        onChange={handleImageUpload}
        className="hidden"
      />

      <Input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type your query..."
        className="flex-1 border-0 focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent"
      />

      <Button
        onClick={handleSearch}
        disabled={isLoading || (!text && !image)}
        size="icon"
        variant="ghost"
        className="rounded-full hover:bg-muted"
      >
        <Send className="h-5 w-5" />
      </Button>
    </div>
  );
}
