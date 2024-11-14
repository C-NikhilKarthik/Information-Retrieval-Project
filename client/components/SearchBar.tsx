import { useState } from "react";
import { Paperclip, Send, X } from "lucide-react"; // Import X icon for close button
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

interface SearchBarProps {
  className?: string;
  onSearchComplete?: (result: any) => void;
  setData: (data: any) => void;
}

export function SearchBar({
  className,
  onSearchComplete,
  setData,
}: SearchBarProps) {
  const [text, setText] = useState("");
  const [image, setImage] = useState<string>("");
  const [imagePreview, setImagePreview] = useState<string>(""); // New state for image preview
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const relativePath = `/Users/nikhilkarthik/Desktop/IIIT Dharwad/7th Sem/IR/Files/CBAM Convolutional Block Attention Module/${file.name}`; // Assuming images are in 'public/images/' directory

      setImage(URL.createObjectURL(file)); // Set the file in the state
      setImagePreview(relativePath); // Generate a preview URL for the image
      toast({
        title: "Image selected",
        description: `Selected: ${file.name}`,
      });
    }
  };

  const removeImage = () => {
    setImage("");
    setImagePreview("");
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

    let queryType = "text";
    if (image && !text) queryType = "image";
    if (image && text) queryType = "combined";

    let endpoint = "";
    const baseUrl = "http://localhost:8000";

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
      const payload = {
        text_input: text || "",
        image_input: imagePreview || "", // Send image preview URL if available
      };

      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Search failed");

      const result = await response.json();
      setData(result?.Result);

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
        "flex flex-col items-center w-full mx-auto p-2 bg-background  shadow-sm",
        className
      )}
    >
      {/* Image Preview */}
      {image && (
        <div className="relative mb-2 w-full max-w-xs">
          <img
            src={image}
            alt="Uploaded Preview"
            className="w-full h-auto rounded border"
          />
          <button
            onClick={removeImage}
            className="absolute top-1 right-1 p-1 bg-red-600 text-white rounded-full hover:bg-red-700"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      <div className="flex items-center gap-2 w-full border rounded-full p-2">
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
          className="flex-1 border-0 focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent shadow-none"
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
    </div>
  );
}
