import { useEffect, useState } from "react";

export function useData() {
    const [data, setData] = useState(null);
    const callApi = async (message: string) => {
        fetch(`http://localhost:3000/api/chat?message=${message}`)
         .then(response => response.json())
         .then(json => {
            setData(json);
            console.log(json);
        });
    };
    return {data, callApi };
  }