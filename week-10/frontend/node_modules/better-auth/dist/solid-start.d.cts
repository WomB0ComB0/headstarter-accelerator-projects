declare function toSolidStartHandler(auth: {
    handler: (request: Request) => Promise<Response>;
} | ((request: Request) => Promise<Response>)): {
    GET: (event: {
        request: Request;
    }) => Promise<Response>;
    POST: (event: {
        request: Request;
    }) => Promise<Response>;
};

export { toSolidStartHandler };
